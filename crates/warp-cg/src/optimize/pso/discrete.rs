use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;

use crate::optimize::{
    DiscreteEvaluationRecord, DiscreteObjectiveEvaluator, ObjectiveEvaluation, ObjectiveEvaluator,
    ParameterBound,
};

pub(super) fn discrete_probability_bounds(choice_counts: &[usize]) -> Vec<ParameterBound> {
    choice_counts
        .iter()
        .enumerate()
        .flat_map(|(variable_index, choice_count)| {
            (0..*choice_count).map(move |choice_index| ParameterBound {
                name: format!("discrete_{variable_index}_{choice_index}_probability"),
                min: 0.0,
                max: 1.0,
            })
        })
        .collect()
}

pub(super) struct DiscreteProbabilityEvaluator<'a> {
    choice_counts: Vec<usize>,
    evaluator: &'a mut dyn DiscreteObjectiveEvaluator,
    rng: ChaCha12Rng,
    use_dilation: bool,
    dilation_alpha: f64,
    pub(super) evaluations: Vec<DiscreteEvaluationRecord>,
    pub(super) best_choices: Vec<usize>,
    pub(super) best_value: f64,
}

impl<'a> DiscreteProbabilityEvaluator<'a> {
    pub(super) fn new(
        choice_counts: &[usize],
        evaluator: &'a mut dyn DiscreteObjectiveEvaluator,
        seed: u64,
        use_dilation: bool,
        dilation_alpha: f64,
    ) -> Self {
        Self {
            choice_counts: choice_counts.to_vec(),
            evaluator,
            rng: ChaCha12Rng::seed_from_u64(seed),
            use_dilation,
            dilation_alpha,
            evaluations: Vec::new(),
            best_choices: Vec::new(),
            best_value: f64::INFINITY,
        }
    }

    fn sample_choices(&mut self, probabilities: &[f64]) -> Vec<usize> {
        sample_choice_indices(
            probabilities,
            &self.choice_counts,
            self.use_dilation,
            self.dilation_alpha,
            &mut self.rng,
        )
    }
}

impl ObjectiveEvaluator for DiscreteProbabilityEvaluator<'_> {
    fn evaluate(&mut self, parameters: &[f64]) -> ObjectiveEvaluation {
        let choices = self.sample_choices(parameters);
        let evaluation = self.evaluator.evaluate_discrete(&choices);
        self.record_evaluation(parameters, choices, &evaluation);
        evaluation
    }

    fn evaluate_batch(&mut self, parameter_sets: &[Vec<f64>]) -> Vec<ObjectiveEvaluation> {
        let choice_sets = parameter_sets
            .iter()
            .map(|parameters| self.sample_choices(parameters))
            .collect::<Vec<_>>();
        let evaluations = self.evaluator.evaluate_discrete_batch(&choice_sets);
        for ((parameters, choices), evaluation) in parameter_sets
            .iter()
            .zip(choice_sets.into_iter())
            .zip(evaluations.iter())
        {
            self.record_evaluation(parameters, choices, evaluation);
        }
        evaluations
    }
}

impl DiscreteProbabilityEvaluator<'_> {
    fn record_evaluation(
        &mut self,
        probabilities: &[f64],
        choices: Vec<usize>,
        evaluation: &ObjectiveEvaluation,
    ) {
        let penalty = if self.best_value.is_finite() {
            self.best_value * 10.0
        } else {
            1.0e12
        };
        let objective = evaluation.objective_or_penalty(penalty);
        if objective < self.best_value {
            self.best_value = objective;
            self.best_choices = choices.clone();
        }
        self.evaluations.push(DiscreteEvaluationRecord {
            iteration: self.evaluations.len(),
            objective,
            choices,
            probabilities: probabilities.to_vec(),
        });
    }
}

pub(super) fn sample_choice_indices(
    probabilities: &[f64],
    choice_counts: &[usize],
    use_dilation: bool,
    dilation_alpha: f64,
    rng: &mut ChaCha12Rng,
) -> Vec<usize> {
    let mut offset = 0usize;
    let mut sampled = Vec::with_capacity(choice_counts.len());
    for &choice_count in choice_counts {
        let segment = &probabilities[offset..offset + choice_count];
        sampled.push(sample_segment(segment, use_dilation, dilation_alpha, rng));
        offset += choice_count;
    }
    sampled
}

fn sample_segment(
    segment: &[f64],
    use_dilation: bool,
    dilation_alpha: f64,
    rng: &mut ChaCha12Rng,
) -> usize {
    let mut weights = segment
        .iter()
        .map(|value| value.max(0.0))
        .collect::<Vec<_>>();
    normalize_weights(&mut weights);
    if weights.iter().all(|weight| *weight == 0.0) {
        return rng.gen_range(0..segment.len());
    }
    if use_dilation {
        dilate_probabilities(&mut weights, dilation_alpha);
    }
    let total = weights.iter().sum::<f64>();
    let mut draw = rng.gen::<f64>() * total;
    for (index, weight) in weights.iter().enumerate() {
        draw -= weight;
        if draw <= 0.0 {
            return index;
        }
    }
    segment.len().saturating_sub(1)
}

fn normalize_weights(weights: &mut [f64]) {
    let total = weights.iter().sum::<f64>();
    if total <= 0.0 {
        weights.fill(0.0);
        return;
    }
    for weight in weights {
        *weight /= total;
    }
}

pub(super) fn dilate_probabilities(probabilities: &mut [f64], alpha: f64) {
    for probability in probabilities.iter_mut() {
        let scaled = (*probability * 2.0).powf(alpha);
        *probability = 1.0 - 1.0 / (1.0 + scaled);
    }
    normalize_weights(probabilities);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discrete_probability_bounds_expand_choice_counts() {
        let bounds = discrete_probability_bounds(&[2, 3]);

        assert_eq!(bounds.len(), 5);
        assert_eq!(bounds[0].name, "discrete_0_0_probability");
        assert!(bounds.iter().all(|bound| bound.min == 0.0));
        assert!(bounds.iter().all(|bound| bound.max == 1.0));
    }

    #[test]
    fn sample_choice_indices_uses_normalized_probability_segments() {
        let mut rng = ChaCha12Rng::seed_from_u64(7);
        let choices =
            sample_choice_indices(&[0.0, 1.0, 0.0, 0.0, 9.0], &[2, 3], false, 8.0, &mut rng);

        assert_eq!(choices, vec![1, 2]);
    }

    #[test]
    fn dilation_sharpens_probability_distribution_like_fst_pso_smoothramp() {
        let mut probabilities = vec![0.25, 0.75];
        dilate_probabilities(&mut probabilities, 8.0);

        assert!(probabilities[0] < 0.01);
        assert!(probabilities[1] > 0.99);
        assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);
    }
}
