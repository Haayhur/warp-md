use geo::{
    algorithm::{bool_ops::FillRule, unary_union},
    Area, BooleanOps, Buffer, Coord, LineString, MultiPolygon, Polygon, SimplifyVwPreserve,
};

use crate::build_layout::LayoutBounds;

use super::super::{
    forward_rotated_xy, layout_bounds_polygon, polygon_area, signed_polygon_area,
    transformed_polygon_points, LeafletRegion, RegionGeometry,
};

const CURVE_BOOLEAN_SEGMENTS: usize = 256;
const SIMPLIFY_EPSILON_ANGSTROM: f64 = 1.0e-7;

pub(in crate::build_contract) fn geo_simple_polygon_union_area(
    polygons: &[Vec<[f32; 2]>],
    bounds: LayoutBounds,
) -> Option<f32> {
    let bounds_polygon = polygon_from_ring(&layout_bounds_polygon(bounds))?;
    let clipped = clipped_multipolygon(polygons_to_multipolygon(polygons)?, &bounds_polygon);
    Some(area_f32(&clipped))
}

pub(in crate::build_contract) fn geo_nested_polygon_inset_area(
    outer: &[[f32; 2]],
    holes: &[Vec<[f32; 2]>],
    inset_angstrom: f32,
) -> Option<f32> {
    let mut outer_geometry = MultiPolygon(vec![polygon_from_ring(outer)?]);
    if inset_angstrom > 1.0e-6 {
        outer_geometry = outer_geometry
            .buffer(-(inset_angstrom as f64))
            .simplify_vw_preserve(SIMPLIFY_EPSILON_ANGSTROM);
    }
    if holes.is_empty() {
        return Some(area_f32(&outer_geometry));
    }
    let mut hole_geometry = polygons_to_multipolygon(holes)?;
    if inset_angstrom > 1.0e-6 {
        hole_geometry = hole_geometry
            .buffer(inset_angstrom as f64)
            .simplify_vw_preserve(SIMPLIFY_EPSILON_ANGSTROM);
    }
    Some(area_f32(&outer_geometry.difference_with_fill_rule(
        &hole_geometry,
        FillRule::NonZero,
    )))
}

pub(in crate::build_contract) fn geo_nested_polygon_forest_inset_area(
    rings: &[Vec<[f32; 2]>],
    depths: &[usize],
    inset_angstrom: f32,
) -> Option<f32> {
    if rings.is_empty() || rings.len() != depths.len() {
        return None;
    }
    let mut solid_polygons = Vec::new();
    let mut hole_polygons = Vec::new();
    for (ring, depth) in rings.iter().zip(depths.iter()) {
        let polygon = polygon_from_ring(ring)?;
        if depth % 2 == 0 {
            solid_polygons.push(polygon);
        } else {
            hole_polygons.push(polygon);
        }
    }
    let solids = unary_union(&solid_polygons);
    let mut geometry = if hole_polygons.is_empty() {
        solids
    } else {
        solids.difference_with_fill_rule(&unary_union(&hole_polygons), FillRule::NonZero)
    };
    if inset_angstrom > 1.0e-6 {
        geometry = geometry
            .buffer(-(inset_angstrom as f64))
            .simplify_vw_preserve(SIMPLIFY_EPSILON_ANGSTROM);
    }
    Some(area_f32(&geometry))
}

pub(in crate::build_contract) fn geo_polygonized_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    let bounds_polygon = polygon_from_ring(&layout_bounds_polygon(bounds))?;
    let mut clipped_regions = Vec::new();
    for region in regions {
        let clipped = clipped_multipolygon(polygonized_region(region)?, &bounds_polygon);
        if clipped.0.is_empty() {
            continue;
        }
        clipped_regions.push(clipped);
    }
    Some(area_f32(&unary_union(&clipped_regions)))
}

fn clipped_multipolygon(
    geometry: MultiPolygon<f64>,
    bounds_polygon: &Polygon<f64>,
) -> MultiPolygon<f64> {
    geometry
        .intersection(bounds_polygon)
        .simplify_vw_preserve(SIMPLIFY_EPSILON_ANGSTROM)
}

fn polygons_to_multipolygon(polygons: &[Vec<[f32; 2]>]) -> Option<MultiPolygon<f64>> {
    let mut prepared = Vec::new();
    for polygon in polygons {
        prepared.push(polygon_from_ring(polygon)?);
    }
    Some(unary_union(&prepared))
}

fn polygonized_region(region: &LeafletRegion) -> Option<MultiPolygon<f64>> {
    let ring = match &region.geometry {
        RegionGeometry::Circle {
            center_angstrom,
            radius_angstrom,
        } => circle_ring(*center_angstrom, *radius_angstrom),
        RegionGeometry::Ellipse {
            center_angstrom,
            radius_angstrom,
            rotate_degrees,
        } => ellipse_ring(*center_angstrom, *radius_angstrom, *rotate_degrees),
        RegionGeometry::Rectangle {
            center_angstrom,
            size_angstrom,
            rotate_degrees,
        } => rectangle_ring(*center_angstrom, *size_angstrom, *rotate_degrees),
        RegionGeometry::Polygon { .. } => transformed_polygon_points(region),
    };
    Some(MultiPolygon(vec![polygon_from_ring(&ring)?]))
}

fn circle_ring(center: [f32; 2], radius: f32) -> Vec<[f32; 2]> {
    if radius <= 0.0 {
        return Vec::new();
    }
    (0..CURVE_BOOLEAN_SEGMENTS)
        .map(|idx| {
            let theta = std::f32::consts::TAU * idx as f32 / CURVE_BOOLEAN_SEGMENTS as f32;
            [
                center[0] + radius * theta.cos(),
                center[1] + radius * theta.sin(),
            ]
        })
        .collect()
}

fn ellipse_ring(center: [f32; 2], radius: [f32; 2], rotate_degrees: f32) -> Vec<[f32; 2]> {
    if radius[0] <= 0.0 || radius[1] <= 0.0 {
        return Vec::new();
    }
    (0..CURVE_BOOLEAN_SEGMENTS)
        .map(|idx| {
            let theta = std::f32::consts::TAU * idx as f32 / CURVE_BOOLEAN_SEGMENTS as f32;
            forward_rotated_xy(
                [radius[0] * theta.cos(), radius[1] * theta.sin()],
                center,
                rotate_degrees,
            )
        })
        .collect()
}

fn rectangle_ring(center: [f32; 2], size: [f32; 2], rotate_degrees: f32) -> Vec<[f32; 2]> {
    let half = [size[0] * 0.5, size[1] * 0.5];
    [
        [-half[0], -half[1]],
        [half[0], -half[1]],
        [half[0], half[1]],
        [-half[0], half[1]],
    ]
    .into_iter()
    .map(|point| forward_rotated_xy(point, center, rotate_degrees))
    .collect()
}

fn polygon_from_ring(ring: &[[f32; 2]]) -> Option<Polygon<f64>> {
    if ring.len() < 3 || polygon_area(ring) <= 1.0e-8 {
        return None;
    }
    Some(Polygon::new(closed_linestring(ring)?, Vec::new()))
}

fn closed_linestring(ring: &[[f32; 2]]) -> Option<LineString<f64>> {
    if ring.len() < 3 {
        return None;
    }
    let mut oriented = ring.to_vec();
    if signed_polygon_area(&oriented) < 0.0 {
        oriented.reverse();
    }
    let mut coords = oriented
        .iter()
        .map(|point| Coord {
            x: point[0] as f64,
            y: point[1] as f64,
        })
        .collect::<Vec<_>>();
    if coords.first()? != coords.last()? {
        coords.push(*coords.first()?);
    }
    Some(LineString(coords))
}

fn area_f32(geometry: &MultiPolygon<f64>) -> f32 {
    geometry.unsigned_area().max(0.0) as f32
}
