use traj_core::frame::Box3;

use super::DtDecimation;

pub(super) fn msd_cols(axis: Option<[f64; 3]>, n_types: usize) -> usize {
    let components = if axis.is_some() { 5 } else { 4 };
    components * (n_types + 1)
}

pub(super) fn lag_allowed(lag: usize, dec: Option<DtDecimation>) -> bool {
    if let Some(dec) = dec {
        if lag > dec.cut2 && (lag % dec.stride2) != 0 {
            return false;
        }
        if lag > dec.cut1 && (lag % dec.stride1) != 0 {
            return false;
        }
    }
    true
}

pub(super) fn box_lengths(box_: &Box3) -> Option<[f64; 3]> {
    match box_ {
        Box3::Orthorhombic { lx, ly, lz } => Some([*lx as f64, *ly as f64, *lz as f64]),
        Box3::Triclinic { .. } => None,
        Box3::None => None,
    }
}
