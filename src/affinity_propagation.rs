use std::cmp::Eq;
use std::collections::HashMap;
use std::hash::Hash;

use ndarray::Array2;
use num_traits::Float;

use crate::algorithm::Calculation;
use crate::similarity::Similarity;

/// Implementation derived from sklearn AffinityPropagation implementation
/// https://github.com/scikit-learn/scikit-learn/blob/0d378913b/sklearn/cluster/_affinity_propagation.py#L38
pub struct AffinityPropagation<F> {
    damping: F,
    preference: F,
    threads: usize,
    convergence_iter: usize,
    max_iterations: usize,
    has_converged: bool,
}

impl<F> Default for AffinityPropagation<F>
where
    F: Float + Send + Sync,
{
    /// Create new model with default parameters
    ///
    /// - damping: 0.5
    /// - threads: 4
    /// - max_iterations: 100
    /// - convergence_iter: 10
    /// - preference: -10.
    fn default() -> Self {
        Self {
            damping: F::from(0.5).unwrap(),
            threads: 4,
            max_iterations: 100,
            convergence_iter: 10,
            preference: F::from(-10.).unwrap(),
            has_converged: false,
        }
    }
}

impl<F> AffinityPropagation<F>
where
    F: Float + Send + Sync,
{
    /// Create new model with provided parameters
    ///
    /// - damping: 0 <= damping <= 1
    /// - threads: parallel threads for analysis
    /// - max_iterations: total allowed iterations
    /// - convergence_iter: number of iterations to run before checking for convergence
    /// - preference: non-positive number representing a data point's desire to be its own exemplar
    pub fn new(
        damping: F,
        threads: usize,
        max_iterations: usize,
        convergence_iter: usize,
        preference: F,
    ) -> Self {
        assert!(
            damping >= F::from(0.).unwrap() && damping <= F::from(1.).unwrap(),
            "invalid damping value provided"
        );
        assert!(
            F::from(preference).unwrap() <= F::from(0.).unwrap(),
            "invalid preference provided"
        );
        Self {
            damping,
            threads,
            max_iterations,
            convergence_iter,
            preference,
            has_converged: false,
        }
    }

    /// Generate cluster predictions for set of `x` values and `y` labels
    /// - x: 2-D array of (rows=samples, cols=attr_values)
    /// - y: Slice of label values attached to each row in `x`
    /// - s: Similarity calculator -> must generate an N x N matrix
    pub fn predict<'a, S, L>(
        &mut self,
        x: Array2<F>,
        y: &'a [L],
        s: S,
    ) -> HashMap<&'a L, Vec<&'a L>>
    where
        S: Similarity<F>,
        L: Eq + Hash,
    {
        let s = s.similarity(x);
        let s_dim = s.dim();
        assert_eq!(s_dim.0, s_dim.1, "similarity dim must be NxN");

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.threads)
            .build()
            .unwrap();
        let results = pool.scope(move |_| {
            let mut calculation = Calculation {
                similarity: s,
                responsibility: Array2::zeros(s_dim),
                availability: Array2::zeros(s_dim),
                damping: self.damping,
            };
            calculation.add_preference_to_sim(self.preference);
            for _ in 0..self.convergence_iter {
                calculation.update_r();
                calculation.update_a();
            }
            let mut final_exemplars = calculation.generate_exemplars();
            for _ in self.convergence_iter..self.max_iterations {
                calculation.update_r();
                calculation.update_a();
                let sol_map = calculation.generate_exemplars();
                if final_exemplars.len() == sol_map.len()
                    && final_exemplars.iter().all(|k| sol_map.contains(k))
                {
                    self.has_converged = true;
                    break;
                }
                final_exemplars = sol_map;
            }
            calculation.generate_exemplar_map(final_exemplars)
        });
        HashMap::from_iter(
            results
                .into_iter()
                .map(|(key, value)| (&y[key], value.iter().map(|v| &y[*v]).collect::<Vec<&L>>())),
        )
    }

    pub fn converged(&self) -> bool {
        self.has_converged
    }
}

#[cfg(test)]
mod test {
    use ndarray::{arr2, Array2};

    use crate::{AffinityPropagation, Euclidean};

    #[test]
    fn simple() {
        let x: Array2<f32> = arr2(&[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]);
        let y = vec!["1", "2", "3"];
        let mut ap = AffinityPropagation::default();
        let results = ap.predict(x, &y, Euclidean::default());
        assert!(results.len() == 1 && results.contains_key(&"2"));
    }
}
