use ndarray::{Array2, Axis};
use num_traits::Float;

pub trait Similarity<F>
where
    F: Float + Send + Sync,
{
    fn similarity(self, x: Array2<F>) -> Array2<F>;
}

pub struct Euclidean;

impl Default for Euclidean {
    fn default() -> Self {
        Euclidean {}
    }
}

impl<F> Similarity<F> for Euclidean
where
    F: Float + Send + Sync,
{
    /// Row-by-row similarity calculation using negative euclidean distance
    fn similarity(self, x: Array2<F>) -> Array2<F> {
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
