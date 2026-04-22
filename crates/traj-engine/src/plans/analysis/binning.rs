use traj_core::error::{TrajError, TrajResult};

pub(crate) fn resolve_bins(
    explicit: Option<usize>,
    bin: f64,
    extent: f64,
    owner: &'static str,
) -> TrajResult<usize> {
    if let Some(value) = explicit {
        return Ok(value.max(1));
    }
    if !extent.is_finite() || extent <= 0.0 {
        return Err(TrajError::Mismatch(format!(
            "{owner} requires positive spatial extent"
        )));
    }
    Ok(((extent / bin).round() as usize).max(1))
}

pub(crate) fn periodic_bin(value: f64, extent: f64, bins: usize) -> Option<usize> {
    if !value.is_finite() || !extent.is_finite() || extent <= 0.0 || bins == 0 {
        return None;
    }
    let wrapped = value.rem_euclid(extent);
    let mut index = ((wrapped / extent) * bins as f64).floor() as usize;
    if index >= bins {
        index = bins - 1;
    }
    Some(index)
}

pub(crate) fn bounded_bin(value: f64, min: f64, max: f64, bins: usize) -> Option<usize> {
    if !value.is_finite() || !min.is_finite() || !max.is_finite() || max <= min || bins == 0 {
        return None;
    }
    if value < min || value > max {
        return None;
    }
    if value == max {
        return Some(bins - 1);
    }
    let frac = (value - min) / (max - min);
    let mut index = (frac * bins as f64).floor() as usize;
    if index >= bins {
        index = bins - 1;
    }
    Some(index)
}

pub(crate) fn build_centers(min: f64, max: f64, bins: usize) -> Vec<f32> {
    if bins == 0 {
        return Vec::new();
    }
    let step = (max - min) / bins as f64;
    let mut out = Vec::with_capacity(bins);
    for idx in 0..bins {
        out.push((min + (idx as f64 + 0.5) * step) as f32);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{bounded_bin, build_centers, periodic_bin, resolve_bins};

    #[test]
    fn periodic_bin_wraps_negative_values() {
        assert_eq!(periodic_bin(-0.1, 2.0, 4), Some(3));
        assert_eq!(periodic_bin(2.0, 2.0, 4), Some(0));
    }

    #[test]
    fn bounded_bin_keeps_upper_edge_in_last_bin() {
        assert_eq!(bounded_bin(1.0, 0.0, 1.0, 4), Some(3));
        assert_eq!(bounded_bin(-0.1, 0.0, 1.0, 4), None);
    }

    #[test]
    fn build_centers_places_midpoints() {
        assert_eq!(build_centers(0.0, 2.0, 4), vec![0.25, 0.75, 1.25, 1.75]);
    }

    #[test]
    fn resolve_bins_reports_owner_context() {
        let error = resolve_bins(None, 0.1, 0.0, "density_map").unwrap_err();
        assert!(error
            .to_string()
            .contains("density_map requires positive spatial extent"));
    }
}
