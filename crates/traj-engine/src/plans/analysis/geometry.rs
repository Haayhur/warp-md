pub fn distance_sq(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

pub fn cross_product(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn dot_product(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 0.0]
    }
}

pub fn project_point_onto_plane(
    point: [f32; 3],
    plane_center: [f32; 3],
    plane_normal: [f32; 3],
) -> [f32; 3] {
    let v = [
        point[0] - plane_center[0],
        point[1] - plane_center[1],
        point[2] - plane_center[2],
    ];
    let dist = dot_product(v, plane_normal);
    [
        point[0] - dist * plane_normal[0],
        point[1] - dist * plane_normal[1],
        point[2] - dist * plane_normal[2],
    ]
}
