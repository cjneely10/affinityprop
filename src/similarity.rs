use ndarray::{Array2, ArrayView1, Axis, Zip};
use num_traits::Float;

/// Generate an N x N matrix in which each (i,j) index represents the
/// similarity between row i and row j of `x`
pub(crate) fn calculate_similarity<F, S>(x: &Array2<F>, s: S) -> Array2<F>
where
    F: Float + Send + Sync,
    S: Similarity<F>,
{
    let x_dim = x.dim();
    let mut out = Array2::<F>::zeros((x_dim.0, x_dim.0));
    // Calculate values for half of matrix
    x.axis_iter(Axis(0)).enumerate().for_each(|(idx1, row1)| {
        x.axis_iter(Axis(0))
            .enumerate()
            .skip(idx1 + 1)
            .for_each(|(idx2, row2)| {
                out[[idx1, idx2]] = s.similarity(&row1, &row2);
            });
    });
    // Copy over for remaining
    (0..x_dim.0).for_each(|idx1| {
        (idx1 + 1..x_dim.0).for_each(|idx2| out[[idx2, idx1]] = out[[idx1, idx2]]);
    });
    out
}

/// Determine the similarity between two data entries.
///
/// Affinity Propagation expects *s(i,j)* > *s(i, k)* iff x<sub>i</sub> is more similar
/// to x<sub>j</sub> than to x<sub>k</sub>.
pub trait Similarity<F>
where
    F: Float + Send + Sync,
{
    fn similarity(&self, a: &ArrayView1<F>, b: &ArrayView1<F>) -> F;
}

/// Perform similarity calculation as `-1 * sum((row_i - row_j)**2)`
#[derive(Debug, Clone)]
pub struct NegEuclidean<F> {
    neg_one: F,
}

impl<F> Default for NegEuclidean<F>
where
    F: Float + Send + Sync,
{
    fn default() -> Self {
        Self {
            neg_one: F::from(-1.).unwrap(),
        }
    }
}

impl<F> Similarity<F> for NegEuclidean<F>
where
    F: Float + Send + Sync,
{
    fn similarity(&self, a: &ArrayView1<F>, b: &ArrayView1<F>) -> F {
        let mut row_diff = a - b;
        row_diff.map_inplace(|_a| *_a = (*_a).powi(2));
        self.neg_one * row_diff.sum()
    }
}

/// Perform similarity calculation as `-1 * (row_i . row_j)/(|row_i|*|row_j|)`
#[derive(Debug, Clone)]
pub struct NegCosine<F> {
    neg_one: F,
}

impl<F> Default for NegCosine<F>
where
    F: Float + Send + Sync,
{
    fn default() -> Self {
        Self {
            neg_one: F::from(-1.).unwrap(),
        }
    }
}

impl<F> Similarity<F> for NegCosine<F>
where
    F: Float + Send + Sync,
{
    fn similarity(&self, a: &ArrayView1<F>, b: &ArrayView1<F>) -> F {
        let dot_product: F = Zip::from(a).and(b).map_collect(|r1, r2| *r1 * *r2).sum();
        let x_magnitude = a.map(|r| r.powi(2)).sum().sqrt();
        let y_magnitude = b.map(|r| r.powi(2)).sum().sqrt();
        self.neg_one * dot_product / x_magnitude / y_magnitude
    }
}

/// Perform similarity calculation as `sum(log((a - b)**2))`
#[derive(Default, Debug, Clone)]
pub struct LogEuclidean;

impl<F> Similarity<F> for LogEuclidean
where
    F: Float + Send + Sync,
{
    fn similarity(&self, a: &ArrayView1<F>, b: &ArrayView1<F>) -> F {
        let mut row_diff = a - b;
        row_diff.map_inplace(|_a| *_a = (*_a).powi(2).log2());
        row_diff.sum()
    }
}

#[cfg(test)]
mod test {
    use ndarray::{arr2, Zip};

    use crate::similarity::calculate_similarity;
    use crate::{LogEuclidean, NegCosine, NegEuclidean};

    #[test]
    fn euclidean_similarity() {
        let x = arr2(&[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]);
        let s = calculate_similarity(&x, NegEuclidean::default());
        let actual = arr2(&[[0., -3.0, -12.0], [-3.0, 0., -3.0], [-12.0, -3.0, 0.]]);
        Zip::from(&s)
            .and(&actual)
            .for_each(|a: &f64, b: &f64| assert!((a - b).abs() < 1e-4));
    }

    #[test]
    fn cosine_similarity() {
        let x = arr2(&[[3., 2., 0., 5.], [1., 0., 0., 0.]]);
        let s = calculate_similarity(&x, NegCosine::default());
        let actual = arr2(&[[0., -0.4866], [-0.4866, 0.]]);
        Zip::from(&s)
            .and(&actual)
            .for_each(|a: &f64, b: &f64| assert!((a - b).abs() < 1e-4));
    }

    #[test]
    fn log_euclidean_similarity() {
        let x = arr2(&[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]);
        let s = calculate_similarity(&x, LogEuclidean::default());
        let actual = arr2(&[[0., 0., 6.], [0., 0., 0.], [6., 0., 0.]]);
        Zip::from(&s)
            .and(&actual)
            .for_each(|a: &f64, b: &f64| assert!((a - b).abs() < 1e-4));
    }
}
