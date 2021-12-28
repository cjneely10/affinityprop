use std::collections::HashMap;

use ndarray::Array2;
use num_traits::Float;

use crate::algorithm::Calculation;
use crate::similarity::Similarity;

/// Implementation derived from sklearn AffinityPropagation implementation
/// <https://github.com/scikit-learn/scikit-learn/blob/0d378913b/sklearn/cluster/_affinity_propagation.py#L38>
pub struct AffinityPropagation<F> {
    damping: F,
    preference: F,
    threads: usize,
    convergence_iter: usize,
    max_iterations: usize,
}

impl<F> Default for AffinityPropagation<F>
where
    F: Float + Send + Sync,
{
    /// Create new model with default parameters
    ///
    /// - damping: 0.5
    /// - preference: -10.0
    /// - threads: 4
    /// - convergence_iter: 10
    /// - max_iterations: 100
    fn default() -> Self {
        Self {
            damping: F::from(0.5).unwrap(),
            threads: 4,
            max_iterations: 100,
            convergence_iter: 10,
            preference: F::from(-10.0).unwrap(),
        }
    }
}

impl<F> AffinityPropagation<F>
where
    F: Float + Send + Sync,
{
    /// Create new model with provided parameters
    ///
    /// - preference: non-positive number representing a data point's desire to be its own exemplar
    /// - damping: 0 <= damping <= 1
    /// - threads: parallel threads for analysis
    /// - convergence_iter: number of iterations to run before checking for convergence
    /// - max_iterations: total allowed iterations
    pub fn new(
        preference: F,
        damping: F,
        threads: usize,
        convergence_iter: usize,
        max_iterations: usize,
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
        }
    }

    /// Generate cluster predictions for set of `x` values
    /// - x: 2-D array of (rows=samples, cols=attr_values)
    /// - s: Similarity calculator -> must generate an N x N matrix
    ///
    /// Returns:
    ///
    /// - True/False if algorithm converged to a set of exemplars
    /// - Map where K:V are exemplar_index:{member_indices}
    ///
    /// Example:
    ///
    ///     # use ndarray::{arr2, Array2};
    ///     # use affinityprop::{AffinityPropagation, NegEuclidean};
    ///     let x: Array2<f32> = arr2(&[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]);
    ///     let ap = AffinityPropagation::default();
    ///     let (converged, results) = ap.predict(&x, NegEuclidean::default());
    ///     assert!(results.len() == 1 && results.contains_key(&1) && converged);
    pub fn predict<S>(&self, x: &Array2<F>, s: S) -> (bool, HashMap<usize, Vec<usize>>)
    where
        S: Similarity<F>,
    {
        let s = s.similarity(x);
        let s_dim = s.dim();
        assert_eq!(s_dim.0, s_dim.1, "similarity dim must be NxN");

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.threads)
            .build()
            .unwrap();
        pool.scope(move |_| {
            let mut has_converged = false;
            let mut calculation = Calculation::new(self.damping, self.preference, s);
            for _ in 0..self.convergence_iter {
                calculation.update();
            }
            let mut final_exemplars = calculation.generate_exemplars();
            for _ in self.convergence_iter..self.max_iterations {
                calculation.update();
                let sol_map = calculation.generate_exemplars();
                if final_exemplars.len() == sol_map.len()
                    && final_exemplars.iter().all(|k| sol_map.contains(k))
                {
                    has_converged = true;
                    break;
                }
                final_exemplars = sol_map;
            }
            (
                has_converged,
                calculation.generate_exemplar_map(final_exemplars),
            )
        })
    }
}

#[cfg(test)]
mod test {
    use ndarray::{arr2, Array2};

    use crate::{AffinityPropagation, NegEuclidean};

    #[test]
    fn simple() {
        let x: Array2<f32> = arr2(&[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]);
        let ap = AffinityPropagation::default();
        let (converged, results) = ap.predict(&x, NegEuclidean::default());
        assert!(converged && results.len() == 1 && results.contains_key(&1));
    }
}
