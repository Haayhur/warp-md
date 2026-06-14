use super::settings::AcquisitionKind;

pub(super) fn acquisition_score(kind: AcquisitionKind, best: f64, mean: f64, sigma: f64) -> f64 {
    match kind {
        AcquisitionKind::ExpectedImprovement => expected_improvement(best, mean, sigma),
        AcquisitionKind::LogExpectedImprovement => log_expected_improvement(best, mean, sigma),
    }
}

fn expected_improvement(best: f64, mean: f64, sigma: f64) -> f64 {
    if sigma <= 1.0e-12 || !sigma.is_finite() {
        return (best - mean).max(0.0);
    }
    let improvement = best - mean;
    let z = improvement / sigma;
    (improvement * normal_cdf(z) + sigma * normal_pdf(z)).max(0.0)
}

fn log_expected_improvement(best: f64, mean: f64, sigma: f64) -> f64 {
    let ei = expected_improvement(best, mean, sigma);
    if ei > 0.0 && ei.is_finite() {
        return ei.ln();
    }
    let fallback = (best - mean).max(1.0e-300);
    fallback.ln()
}

fn normal_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

fn normal_cdf(x: f64) -> f64 {
    if x < -8.0 {
        return 0.0;
    }
    if x > 8.0 {
        return 1.0;
    }
    let abs_x = x.abs();
    let t = 1.0 / (1.0 + 0.231_641_9 * abs_x);
    let poly = (((1.330_274_429 * t - 1.821_255_978) * t + 1.781_477_937) * t - 0.356_563_782) * t
        + 0.319_381_530;
    let cdf = 1.0 - normal_pdf(abs_x) * poly * t;
    if x >= 0.0 {
        cdf
    } else {
        1.0 - cdf
    }
}
