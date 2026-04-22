pub(super) fn msd_cols(axis: Option<[f64; 3]>, n_types: usize) -> usize {
    let components = if axis.is_some() { 5 } else { 4 };
    components * (n_types + 1)
}
