use ndarray::parallel::prelude::*;
use ndarray::{s, Array1, Array2, ArrayView, ArrayViewMut, Axis, Dim, Zip};
use rayon::ThreadPoolBuilder;

type Value = f32;

const NEG_INF: Value = (-1. as Value) * Value::INFINITY;

pub struct Config {
    pub damping: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self { damping: 0.1 }
    }
}

pub struct AffinityPropagation {
    similarity: Array2<Value>,
    responsibility: Array2<Value>,
    availability: Array2<Value>,
    workers: usize,
    config: Config,
}

impl AffinityPropagation {
    pub fn begin(r: usize, c: usize, w: usize, iterations: usize) {
        let mut ap = AffinityPropagation::new(r, c, w);
        for i in 0..iterations {
            println!("Beginning iteration {}", i);
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
            config: Config::default(),
        }
    }

    fn update_r(&mut self) {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.workers)
            .build()
            .unwrap();
        pool.scope(move |_| {
            let dim = self.availability.dim().clone();
            let mut v = Array2::zeros(dim);

            // v = S + A
            Zip::from(&mut v)
                .and(&self.similarity)
                .and(&self.availability)
                .par_for_each(|v, &s, &a| *v = s + a);

            // np.fill_diagonal(v, -np.inf)
            v.diag_mut().par_map_inplace(|c| *c = NEG_INF);

            // idx_max = np.argmax(v, axis=1)
            // first_max = v[rows, idx_max]
            let mut max: Vec<(usize, Value)> = Vec::new();
            v.axis_iter(Axis(1))
                .into_par_iter()
                .map(|col| AffinityPropagation::max_argmax(col))
                .collect_into_vec(&mut max);
            let idx_max: Array1<usize> = max.iter().map(|t| t.0).collect::<Vec<usize>>().into();
            let first_max: Array1<Value> = max.iter().map(|t| t.1).collect::<Vec<Value>>().into();

            // v[rows, idx_max] = -np.inf
            Zip::from(v.axis_iter_mut(Axis(0)))
                .and(&idx_max)
                .par_for_each(|mut r, &idx| r.slice_mut(s![idx]).fill(NEG_INF));

            // second_max = v[rows, np.argmax(v, axis=1)]
            let mut max: Vec<(usize, Value)> = Vec::new();
            v.axis_iter(Axis(1))
                .into_par_iter()
                .map(|col| AffinityPropagation::max_argmax(col))
                .collect_into_vec(&mut max);
            let second_max: Array1<Value> = max.iter().map(|t| t.1).collect::<Vec<Value>>().into();

            // max_matrix = np.zeros_like(R) + first_max[:, None]
            let mut max_matrix: Array2<Value> =
                Array2::<Value>::zeros(self.responsibility.dim()) + first_max.insert_axis(Axis(1));
            // max_matrix[rows, idx_max] = second_max
            Zip::from(max_matrix.axis_iter_mut(Axis(0)))
                .and(&idx_max)
                .and(&second_max)
                .par_for_each(|mut r, &idx, &max| r.slice_mut(s![idx]).fill(max));

            // new_val = S - max_matrix
            let mut new_val = Array2::<Value>::zeros(self.similarity.dim());
            Zip::from(&mut new_val)
                .and(&self.responsibility)
                .and(&max_matrix)
                .par_for_each(|n, &r, &m| *n = r - m);

            // R = R * damping + (1 - damping) * new_val
            let damping = self.config.damping;
            Zip::from(&mut self.responsibility)
                .and(&new_val)
                .par_for_each(|r, &n| *r = *r * damping + (1. - damping) * n);
        });
    }

    fn update_a(&mut self) {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.workers)
            .build()
            .unwrap();
        pool.scope(move |_| {
            let dim = self.availability.dim().clone();
            self.responsibility = self.responsibility.clone();
        });
    }

    fn max_argmax(data: ArrayView<Value, Dim<[usize; 1]>>) -> (usize, Value) {
        let mut max_pos = 0;
        let mut max: Value = 0.;
        data.iter()
            .enumerate()
            .map(|(idx, val)| {
                if *val > max {
                    max = *val;
                    max_pos = idx;
                }
            })
            .last();
        (max_pos, max)
    }

    // TODO: May be of different types, need to see...
    fn similarity(x: Value, y: Value) {}
}

#[cfg(test)]
mod test {
    use crate::affinity_propagation::AffinityPropagation;

    #[test]
    fn init() {
        AffinityPropagation::begin(100000, 100000, 16, 10);
    }
}
