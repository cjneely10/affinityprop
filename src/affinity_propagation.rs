use ndarray::parallel::prelude::*;
use ndarray::{array, Array, Array2, Axis, Zip};
use rayon::ThreadPoolBuilder;

type Value = f32;

pub struct AffinityPropagation {
    similarity: Array2<Value>,
    responsibility: Array2<Value>,
    availability: Array2<Value>,
    workers: usize,
}

impl AffinityPropagation {
    pub fn begin(r: usize, c: usize, w: usize, iterations: usize) {
        let mut ap = AffinityPropagation::new(r, c, w);
        for _ in 0..iterations {
            ap.update_r();
            ap.update_a();
        }
    }

    fn new(r: usize, c: usize, w: usize) -> Self {
        Self {
            similarity: Array2::zeros((r, c)),
            responsibility: Array2::zeros((r, c)),
            availability: Array2::zeros((r, c)),
            workers: w,
        }
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
                .par_for_each(|v, &s, &a| *v = s + a);

            // rows = np.arange(x.shape[0])
            let rows = Array::range(0., dim.0 as f32, 1.);

            // np.fill_diagonal(v, -np.inf)
            v.diag_mut().par_map_inplace(|c| *c = -1. * f32::INFINITY);

            // idx_max = np.argmax(v, axis=1)
            // first_max = v[rows, idx_max]

            // v[rows, idx_max] = -np.inf
            // second_max = v[rows, np.argmax(v, axis=1)]

            // max_matrix = np.zeros_like(R) + first_max[:, None]
            // max_matrix[rows, idx_max] = second_max

            // new_val = S - max_matrix

            // R = R * damping + (1 - damping) * new_val
        });
    }

    fn update_a(&mut self) {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.workers)
            .build()
            .unwrap();
        pool.scope(move |s| {
            let dim = self.availability.dim().clone();
        });
    }
}

#[cfg(test)]
mod test {
    use crate::affinity_propagation::AffinityPropagation;

    #[test]
    fn init() {
        AffinityPropagation::begin(10, 10, 5, 2);
    }
}
