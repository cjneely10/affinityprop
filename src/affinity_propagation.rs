use ndarray::parallel::prelude::*;
use ndarray::{array, Array, Array2, Axis, Zip};
use ndarray_stats::QuantileExt;
use rayon::ThreadPoolBuilder;

type Value = f32;

pub struct AffinityPropagation {
    similarity: Array2<Value>,
    responsibility: Array2<Value>,
    availability: Array2<Value>,
    workers: usize,
}

impl AffinityPropagation {
    pub fn begin() {
        todo!();
    }

    fn update_r(&mut self) {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.workers)
            .build()
            .unwrap();
        pool.scope(move |s| {
            let dim = self.availability.dim().clone();
            let mut v = Array2::zeros(dim);

            // v = S + A
            Zip::from(&mut v)
                .and(&self.similarity)
                .and(&self.availability)
                .par_for_each(|v, &s, &a| {
                    *v = s + a;
                });

            // rows = np.arange(x.shape[0])
            let rows = Array::range(0., dim.0 as f32, 1.);

            // np.fill_diagonal(v, -np.inf)
            v.diag_mut().par_map_inplace(|c| *c = -1. * f32::INFINITY);

            // idx_max = np.argmax(v, axis=1)
            // first_max = v[rows, idx_max]
        });
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn init() {}
}
