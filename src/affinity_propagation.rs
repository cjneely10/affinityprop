use std::collections::HashMap;

use ndarray::Array2;
use num_traits::Float;

use crate::algorithm::APAlgorithm;
use crate::similarity::{calculate_similarity, Similarity};
use crate::Preference;

/// Index of data point in test data
pub type Idx = usize;
/// Cluster of data indices
pub type Cluster = Vec<Idx>;
/// Map of exemplar index to cluster
pub type ClusterMap = HashMap<Idx, Cluster>;
/// Packaged results, including algorithm convergence status
pub type ClusterResults = (bool, ClusterMap);

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
///     # use affinityprop::{AffinityPropagation, NegEuclidean, Preference};
///     let x: Array2<f32> = arr2(&[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]);
///     let ap = AffinityPropagation::default();
///     let (converged, results) = ap.predict(&x, NegEuclidean::default(), Preference::Value(-10.));
///     assert!(converged && results.len() == 1 && results.contains_key(&1));
#[derive(Debug, Clone)]
pub struct AffinityPropagation<F> {
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
    /// - damping: 0.5
    /// - threads: 4
    /// - convergence_iter: 10
    /// - max_iterations: 100
    fn default() -> Self {
        Self {
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
    /// - damping: 0 < damping < 1
    /// - threads: parallel threads for analysis
    /// - convergence_iter: number of iterations to run before checking for convergence
    /// - max_iterations: total allowed iterations
    pub fn new(damping: F, threads: usize, convergence_iter: usize, max_iterations: usize) -> Self {
        assert!(
            damping > F::from(0.).unwrap() && damping < F::from(1.).unwrap(),
            "invalid damping value provided"
        );
        Self {
            damping,
            threads,
            max_iterations,
            convergence_iter,
        }
    }

    /// Generate cluster predictions for set of `x` values
    /// - x: 2-D array of (rows=samples, cols=attr_values)
    /// - s: Similarity calculator
    /// - p: Median value, provided value, or array of preference values
    ///
    /// Results will be calculated using the floating-point precision defined
    /// by the input data
    ///
    /// Returns:
    ///
    /// - True/False if algorithm converged to a set of exemplars
    /// - Map where K:V are exemplar_index:{member_indices}
    pub fn predict<S>(&self, x: &Array2<F>, s: S, p: Preference<F>) -> ClusterResults
    where
        S: Similarity<F>,
    {
        let s = calculate_similarity(x, s);
        assert!(s.is_square(), "similarity dim must be NxN");
        self.predict_parallel(s, p)
    }

    /// Generate cluster predictions for a pre-calculated similarity matrix
    /// - x: 2-D square similarity matrix
    /// - p: Median value, provided value, or array of preference values
    ///
    /// Results will be calculated using the floating-point precision defined
    /// by the input data
    ///
    /// Returns:
    ///
    /// - True/False if algorithm converged to a set of exemplars
    /// - Map where K:V are exemplar_index:{member_indices}
    pub fn predict_precalculated(&self, s: Array2<F>, p: Preference<F>) -> ClusterResults {
        assert!(s.is_square(), "similarity dim must be NxN");
        self.predict_parallel(s, p)
    }

    fn predict_parallel(&self, s: Array2<F>, p: Preference<F>) -> ClusterResults {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.threads)
            .build()
            .unwrap();
        pool.scope(move |_| {
            let mut calculation = APAlgorithm::new(self.damping, p, s);
            calculation.predict(self.convergence_iter, self.max_iterations)
        })
    }
}

#[cfg(test)]
mod test {
    use ndarray::{arr2, Array2};

    use crate::similarity::calculate_similarity;
    use crate::Preference::Value;
    use crate::{AffinityPropagation, NegCosine, NegEuclidean, Preference};

    fn test_data() -> Array2<f32> {
        arr2(&[
            [3., 4., 3., 2., 1.],
            [4., 3., 5., 1., 1.],
            [3., 5., 3., 3., 3.],
            [2., 1., 3., 3., 2.],
            [1., 1., 3., 2., 3.],
        ])
    }

    #[test]
    fn simple() {
        let x: Array2<f32> = arr2(&[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]);
        let ap = AffinityPropagation::default();
        let (converged, results) = ap.predict(&x, NegEuclidean::default(), Preference::Value(-10.));
        assert!(converged);
        assert_eq!(1, results.len());
        assert!(results.contains_key(&1));
    }

    #[test]
    fn zero() {
        let x: Array2<f32> = arr2(&[[]]);
        let ap = AffinityPropagation::default();
        let (converged, _) = ap.predict(&x, NegCosine::default(), Value(-10.));
        assert!(!converged);
    }

    #[test]
    fn one() {
        let x: Array2<f32> = arr2(&[[0., 1., 0.]]);
        let ap = AffinityPropagation::default();
        let (converged, _) = ap.predict(&x, NegCosine::default(), Value(-10.));
        assert!(!converged);
    }

    #[test]
    fn with_cosine() {
        let x: Array2<f32> = arr2(&[[0., 1., 0.], [2., 3., 2.], [3., 2., 3.]]);
        let ap = AffinityPropagation::default();
        let (converged, results) = ap.predict(&x, NegCosine::default(), Value(-10.));
        assert!(converged);
        assert_eq!(1, results.len());
        assert!(results.contains_key(&0));
    }

    #[test]
    fn with_cosine_unconverged() {
        let x: Array2<f32> = arr2(&[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]);
        let ap = AffinityPropagation::default();
        let (converged, _) = ap.predict(&x, NegCosine::default(), Value(-10.));
        assert!(!converged);
    }

    #[test]
    fn with_parameters() {
        let x: Array2<f32> = arr2(&[[1., 2., 1.], [2., 3., 2.], [3., 2., 3.]]);
        let ap = AffinityPropagation::new(0.5, 2, 10, 100);
        let (converged, results) = ap.predict(&x, NegCosine::default(), Preference::Median);
        assert!(converged);
        assert_eq!(2, results.len());
        assert!(results.contains_key(&0) && results.contains_key(&2));
    }

    #[test]
    fn larger() {
        let x: Array2<f32> = test_data();
        let ap = AffinityPropagation::default();
        let (converged, results) = ap.predict(&x, NegEuclidean::default(), Value(-10.));
        assert!(converged);
        assert_eq!(1, results.len());
        assert!(results.contains_key(&0));
    }

    #[test]
    fn test_pre_calculated() {
        let x: Array2<f32> = test_data();
        let s = calculate_similarity(&x, NegEuclidean::default());
        let ap = AffinityPropagation::default();
        let (converged, results) = ap.predict_precalculated(s, Value(-10.));
        assert!(converged);
        assert_eq!(1, results.len());
        assert!(results.contains_key(&0));
    }
}
