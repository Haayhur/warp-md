pub fn next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005u64).wrapping_add(1);
    *state
}

pub fn next_f64(state: &mut u64) -> f64 {
    let bits = next_u64(state) >> 11;
    (bits as f64) * (1.0 / ((1u64 << 53) as f64))
}

pub fn gaussian_pair(state: &mut u64) -> (f64, f64) {
    let mut u1 = next_f64(state);
    if u1 <= 0.0 {
        u1 = 1e-12;
    }
    let u2 = next_f64(state);
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = std::f64::consts::TAU * u2;
    (r * theta.cos(), r * theta.sin())
}
