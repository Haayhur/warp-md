use super::*;

impl GistDirectPlan {
    fn pair_energy(
        &self,
        chunk: &FrameChunk,
        frame_base: usize,
        atom_i: usize,
        atom_j: usize,
        pbc: GistPbc,
    ) -> TrajResult<f64> {
        if atom_i == atom_j {
            return Ok(0.0);
        }
        if atom_i >= chunk.n_atoms || atom_j >= chunk.n_atoms {
            return Err(TrajError::Mismatch(
                "gist atom index out of bounds for frame".into(),
            ));
        }
        if atom_i >= self.charges.len() || atom_j >= self.charges.len() {
            return Err(TrajError::Mismatch(
                "gist atom index out of nonbonded parameter bounds".into(),
            ));
        }
        let pi = chunk.coords[frame_base + atom_i];
        let pj = chunk.coords[frame_base + atom_j];
        let mut dx = (pi[0] as f64 - pj[0] as f64) * self.length_scale;
        let mut dy = (pi[1] as f64 - pj[1] as f64) * self.length_scale;
        let mut dz = (pi[2] as f64 - pj[2] as f64) * self.length_scale;
        match pbc {
            GistPbc::None => {}
            GistPbc::Orthorhombic { lx, ly, lz } => {
                pbc_math::apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
            }
            GistPbc::Triclinic { cell, inv } => {
                pbc_math::apply_pbc_triclinic(&mut dx, &mut dy, &mut dz, &cell, &inv);
            }
        }
        let r2 = dx * dx + dy * dy + dz * dz;
        if r2 <= 0.0 {
            return Ok(0.0);
        }
        let r = r2.sqrt();
        if r > self.cutoff {
            return Ok(0.0);
        }

        let key = pair_key(atom_i, atom_j);
        let (qprod, sigma, epsilon) = if let Some(pair) = self.exceptions.get(&key) {
            (pair.qprod, pair.sigma, pair.epsilon)
        } else {
            let qprod = self.charges[atom_i] * self.charges[atom_j];
            let sigma = 0.5 * (self.sigmas[atom_i] + self.sigmas[atom_j]);
            let epsilon = (self.epsilons[atom_i] * self.epsilons[atom_j]).sqrt();
            (qprod, sigma, epsilon)
        };

        let mut e = 0.0;
        if epsilon != 0.0 && sigma != 0.0 {
            let sr = sigma / r;
            let sr2 = sr * sr;
            let sr6 = sr2 * sr2 * sr2;
            e += 4.0 * epsilon * (sr6 * sr6 - sr6);
        }
        if qprod != 0.0 {
            e += COULOMB_CONST * qprod / r;
        }
        Ok(e)
    }

    pub(super) fn group_energy(
        &self,
        chunk: &FrameChunk,
        frame: usize,
        group_a: &[u32],
        group_b: &[u32],
        pbc: GistPbc,
    ) -> TrajResult<f64> {
        let frame_base = frame * chunk.n_atoms;
        let mut e_total = 0.0;
        for &ai in group_a.iter() {
            let ai = ai as usize;
            for &aj in group_b.iter() {
                let aj = aj as usize;
                if ai == aj {
                    continue;
                }
                e_total += self.pair_energy(chunk, frame_base, ai, aj, pbc)?;
            }
        }
        Ok(e_total)
    }
}
