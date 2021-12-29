use ndarray::{Array2, Axis};
use num_traits::Float;

pub trait Similarity<F>
where
    F: Float + Send + Sync,
{
    /// Generate an N x N matrix in which each (i,j) index represents the
    /// similarity between row i and row j of `x`
    fn similarity(&self, x: &Array2<F>) -> Array2<F>;
}

pub struct NegEuclidean;

impl Default for NegEuclidean {
    /// Perform similarity calculation as -1 * sum((row_i - row_j)**2)
    fn default() -> Self {
        NegEuclidean {}
    }
}

impl<F> Similarity<F> for NegEuclidean
where
    F: Float + Send + Sync,
{
    /// Negative euclidean similarity
    ///
    ///     # use ndarray::{arr2, Zip};
    ///     # use affinityprop::{NegEuclidean, Similarity};
    ///
    ///     let x = arr2(&[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]);
    ///     let s = NegEuclidean::default().similarity(&x);
    ///     let actual = arr2(&[[0., -3.0, -12.0], [-3.0, 0., -3.0], [-12.0, -3.0, 0.]]);
    ///     Zip::from(&s)
    ///         .and(&actual)
    ///         .for_each(|a: &f64, b: &f64| assert!((a - b).abs() < 1e-8));
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

#[cfg(test)]
mod test {
    use ndarray::{arr2, Zip};

    use crate::{NegEuclidean, Similarity};

    #[test]
    fn valid_similarity() {
        let x = arr2(&[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]);
        let s = NegEuclidean::default().similarity(&x);
        let actual = arr2(&[[0., -3.0, -12.0], [-3.0, 0., -3.0], [-12.0, -3.0, 0.]]);
        Zip::from(&s)
            .and(&actual)
            .for_each(|a: &f64, b: &f64| assert!((a - b).abs() < 1e-8));
    }
}
