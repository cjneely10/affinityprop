use std::collections::HashMap;

use ndarray::Array2;
use num_traits::Float;

use crate::algorithm::APAlgorithm;
use crate::similarity::{calculate_similarity, Similarity};

/// A model whose parameters will be used to cluster data into exemplars.
///
/// - preference: A number representing a data point's desire to be its own exemplar.
/// - damping: The extent to which the current availability/responsibility is modified in an update.
/// - threads: The number of workers that will operate on the data.
/// - convergence_iter: The number of iterations to complete before checking for convergence.
/// - max_iterations: The total number of iterations to attempt.
///
/// Example:
///
///     # use ndarray::{arr2, Array2};
///     # use affinityprop::{AffinityPropagation, NegEuclidean};
///     let x: Array2<f32> = arr2(&[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]);
///     let ap = AffinityPropagation::default();
///     let (converged, results) = ap.predict(&x, NegEuclidean::default());
///     assert!(converged && results.len() == 1 && results.contains_key(&1));
#[derive(Debug, Clone, PartialOrd, PartialEq)]
pub struct AffinityPropagation<F> {
    preference: Option<F>,
    damping: F,
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
    /// - preference: -10.0
    /// - damping: 0.5
    /// - threads: 4
    /// - convergence_iter: 10
    /// - max_iterations: 100
    fn default() -> Self {
        Self {
            preference: Some(F::from(-10.0).unwrap()),
            damping: F::from(0.5).unwrap(),
            threads: 4,
            convergence_iter: 10,
            max_iterations: 100,
        }
    }
}

impl<F> AffinityPropagation<F>
where
    F: Float + Send + Sync,
{
    /// Create new model with provided parameters
    ///
    /// - preference: Median pairwise similarity if None
    /// - damping: 0 < damping < 1
    /// - threads: parallel threads for analysis
    /// - convergence_iter: number of iterations to run before checking for convergence
    /// - max_iterations: total allowed iterations
    pub fn new(
        preference: Option<F>,
        damping: F,
        threads: usize,
        convergence_iter: usize,
        max_iterations: usize,
    ) -> Self {
        assert!(
            damping > F::from(0.).unwrap() && damping < F::from(1.).unwrap(),
            "invalid damping value provided"
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
    /// - s: Similarity calculator
    ///
    /// Results will be calculated using the floating-point precision defined
    /// by the input data
    ///
    /// Returns:
    ///
    /// - True/False if algorithm converged to a set of exemplars
    /// - Map where K:V are exemplar_index:{member_indices}
    pub fn predict<S>(&self, x: &Array2<F>, s: S) -> (bool, HashMap<usize, Vec<usize>>)
    where
        S: Similarity<F>,
    {
        let s = calculate_similarity(x, s);
        assert!(s.is_square(), "similarity dim must be NxN");

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.threads)
            .build()
            .unwrap();
        pool.scope(move |_| {
            let mut has_converged = false;
            let mut calculation = APAlgorithm::new(self.damping, self.preference, s);
            for _ in 0..self.convergence_iter {
                calculation.update();
            }
            let mut final_exemplars = calculation.generate_exemplars();
            for _ in self.convergence_iter..self.max_iterations {
                calculation.update();
                let sol_map = calculation.generate_exemplars();
                if !sol_map.is_empty()
                    && final_exemplars.len() == sol_map.len()
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

    use crate::{AffinityPropagation, NegCosine, NegEuclidean};

    #[test]
    fn simple() {
        let x: Array2<f32> = arr2(&[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]);
        let ap = AffinityPropagation::default();
        let (converged, results) = ap.predict(&x, NegEuclidean::default());
        assert!(converged);
        assert_eq!(1, results.len());
        assert!(results.contains_key(&1));
    }

    #[test]
    fn zero() {
        let x: Array2<f32> = arr2(&[[]]);
        let ap = AffinityPropagation::default();
        let (converged, _) = ap.predict(&x, NegCosine::default());
        assert!(!converged);
    }

    #[test]
    fn one() {
        let x: Array2<f32> = arr2(&[[0., 1., 0.]]);
        let ap = AffinityPropagation::default();
        let (converged, _) = ap.predict(&x, NegCosine::default());
        assert!(!converged);
    }

    #[test]
    fn with_cosine() {
        let x: Array2<f32> = arr2(&[[0., 1., 0.], [2., 3., 2.], [3., 2., 3.]]);
        let ap = AffinityPropagation::default();
        let (converged, results) = ap.predict(&x, NegCosine::default());
        assert!(converged);
        assert_eq!(1, results.len());
        assert!(results.contains_key(&0));
    }

    #[test]
    fn with_cosine_unconverged() {
        let x: Array2<f32> = arr2(&[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]);
        let ap = AffinityPropagation::default();
        let (converged, _) = ap.predict(&x, NegCosine::default());
        assert!(!converged);
    }

    #[test]
    fn with_parameters() {
        let x: Array2<f32> = arr2(&[[1., 2., 1.], [2., 3., 2.], [3., 2., 3.]]);
        let ap = AffinityPropagation::new(None, 0.5, 2, 10, 100);
        let (converged, results) = ap.predict(&x, NegCosine::default());
        assert!(converged);
        assert_eq!(2, results.len());
        assert!(results.contains_key(&0) && results.contains_key(&2));
    }

    #[test]
    fn larger() {
        let x: Array2<f32> = arr2(&[
            [3., 4., 3., 2., 1.],
            [4., 3., 5., 1., 1.],
            [3., 5., 3., 3., 3.],
            [2., 1., 3., 3., 2.],
            [1., 1., 3., 2., 3.],
        ]);
        let ap = AffinityPropagation::default();
        let (converged, results) = ap.predict(&x, NegEuclidean::default());
        assert!(converged);
        assert_eq!(1, results.len());
        assert!(results.contains_key(&0));
    }
}
