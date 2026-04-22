use traj_core::error::{TrajError, TrajResult};

pub fn validate_selection(selection: &[u32], n_atoms: usize) -> TrajResult<()> {
    for &idx in selection {
        let src = idx as usize;
        if src >= n_atoms {
            return Err(TrajError::Mismatch(format!(
                "selection index {idx} out of bounds for trajectory with {n_atoms} atoms"
            )));
        }
    }
    Ok(())
}

pub fn validate_and_materialize_selection(
    selection: &[u32],
    n_atoms: usize,
) -> TrajResult<Vec<usize>> {
    validate_selection(selection, n_atoms)?;
    Ok(selection.iter().map(|&idx| idx as usize).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn materializes_in_bounds_selection_without_reordering() {
        let selection = [2u32, 0, 2, 1];
        let out = validate_and_materialize_selection(&selection, 3).unwrap();
        assert_eq!(out, vec![2usize, 0, 2, 1]);
    }

    #[test]
    fn rejects_out_of_bounds_selection_with_context() {
        let err = validate_and_materialize_selection(&[0u32, 3], 3).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("selection index 3 out of bounds"));
        assert!(message.contains("3 atoms"));
    }
}
