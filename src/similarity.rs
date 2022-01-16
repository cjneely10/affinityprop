use ndarray::{Array2, Axis, Zip};
use num_traits::Float;

/// Determine the N x N similarity matrix for a collection of data.
pub trait Similarity<F>
where
    F: Float + Send + Sync,
{
    /// Generate an N x N matrix in which each (i,j) index represents the
    /// similarity between row i and row j of `x`
    fn similarity(&self, x: &Array2<F>) -> Array2<F>;
}

/// Perform similarity calculation as `-1 * sum((row_i - row_j)**2)`
///
///     use ndarray::{arr2, Zip};
///     use affinityprop::{NegEuclidean, Similarity};
///
///     let x = arr2(&[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]);
///     let s = NegEuclidean::default().similarity(&x);
///     let actual = arr2(&[[0., -3.0, -12.0], [-3.0, 0., -3.0], [-12.0, -3.0, 0.]]);
///     Zip::from(&s)
///         .and(&actual)
///         .for_each(|a: &f64, b: &f64| assert!((a - b).abs() < 1e-8));
#[derive(Debug, Default, Clone)]
pub struct NegEuclidean;

impl<F> Similarity<F> for NegEuclidean
where
    F: Float + Send + Sync,
{
    fn similarity(&self, x: &Array2<F>) -> Array2<F> {
        let x_dim = x.dim();
        let mut out = Array2::<F>::zeros((x_dim.0, x_dim.0));
        let neg_one = F::from(-1.).unwrap();
        x.axis_iter(Axis(0)).enumerate().for_each(|(idx1, row1)| {
            x.axis_iter(Axis(0)).enumerate().for_each(|(idx2, row2)| {
                // Calculate values for half of matrix, copy over for remaining
                if idx2 > idx1 {
                    let mut row_diff = &row1 - &row2;
                    row_diff.map_inplace(|a| *a = (*a).powi(2));
                    out[[idx1, idx2]] = neg_one * row_diff.sum();
                } else {
                    out[[idx1, idx2]] = out[[idx2, idx1]];
                }
            });
        });
        out
    }
}

/// Perform similarity calculation as `-1 * (row_i . row_j)/(|row_i|*|row_j|)`
///
///     use ndarray::{arr2, Zip};
///     use affinityprop::{NegCosine, Similarity};
///
///     let x = arr2(&[[3., 2., 0., 5.], [1., 0., 0., 0.]]);
///     let s = NegCosine::default().similarity(&x);
///     let actual = arr2(&[[0., -0.4866], [-0.4866, 0.]]);
///     Zip::from(&s)
///         .and(&actual)
///         .for_each(|a: &f64, b: &f64| assert!((a - b).abs() < 1e-4));
#[derive(Debug, Default, Clone)]
pub struct NegCosine;

impl<F> Similarity<F> for NegCosine
where
    F: Float + Send + Sync,
{
    fn similarity(&self, x: &Array2<F>) -> Array2<F> {
        let x_dim = x.dim();
        let mut out = Array2::<F>::zeros((x_dim.0, x_dim.0));
        let neg_one = F::from(-1.).unwrap();
        x.axis_iter(Axis(0)).enumerate().for_each(|(idx1, row1)| {
            x.axis_iter(Axis(0)).enumerate().for_each(|(idx2, row2)| {
                // Calculate values for half of matrix, copy over for remaining
                if idx2 > idx1 {
                    let dot_product: F = Zip::from(&row1)
                        .and(&row2)
                        .map_collect(|r1, r2| *r1 * *r2)
                        .sum();
                    let x_magnitude = row1.map(|r| r.powi(2)).sum().sqrt();
                    let y_magnitude = row2.map(|r| r.powi(2)).sum().sqrt();
                    out[[idx1, idx2]] = neg_one * dot_product / x_magnitude / y_magnitude;
                } else {
                    out[[idx1, idx2]] = out[[idx2, idx1]];
                }
            });
        });
        out
    }
}

#[cfg(test)]
mod test {
    use ndarray::{arr2, Zip};

    use crate::{NegCosine, NegEuclidean, Similarity};

    #[test]
    fn euclidean_similarity() {
        let x = arr2(&[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]);
        let s = NegEuclidean::default().similarity(&x);
        let actual = arr2(&[[0., -3.0, -12.0], [-3.0, 0., -3.0], [-12.0, -3.0, 0.]]);
        Zip::from(&s)
            .and(&actual)
            .for_each(|a: &f64, b: &f64| assert!((a - b).abs() < 1e-4));
    }

    #[test]
    fn cosine_similarity() {
        let x = arr2(&[[3., 2., 0., 5.], [1., 0., 0., 0.]]);
        let s = NegCosine::default().similarity(&x);
        let actual = arr2(&[[0., -0.4866], [-0.4866, 0.]]);
        Zip::from(&s)
            .and(&actual)
            .for_each(|a: &f64, b: &f64| assert!((a - b).abs() < 1e-4));
    }
}
