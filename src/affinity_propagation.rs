use ndarray::parallel::prelude::*;
use ndarray::{s, Array1, Array2, ArrayView, Axis, Dim, Zip};
use std::collections::HashSet;
use std::iter::FromIterator;

type Value = f32;

const NEG_INF: Value = (-1. as Value) * Value::INFINITY;

#[derive(Copy, Clone)]
pub struct Config {
    pub damping: f32,
    pub workers: usize,
    pub iterations: usize,
    pub preference: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            damping: 0.5,
            preference: -10.,
            workers: 4,
            iterations: 3,
        }
    }
}

pub trait Similarity {
    fn similarity(self, x: Array2<Value>) -> Array2<Value>;
}

pub struct Euclidean;

impl Similarity for Euclidean {
    fn similarity(self, x: Array2<Value>) -> Array2<Value> {
        let x_dim = x.dim();
        let mut out = Array2::<Value>::zeros((x_dim.0, x_dim.0));
        x.axis_iter(Axis(0)).enumerate().for_each(|(idx1, row1)| {
            x.axis_iter(Axis(0)).enumerate().for_each(|(idx2, row2)| {
                let mut row_diff = &row1 - &row2;
                row_diff.par_mapv_inplace(|a| -1. * a.powi(2));
                out[[idx1, idx2]] = -1. * row_diff.sum();
            });
        });
        out
    }
}

/// Implementation derived from:
/// https://www.ritchievink.com/blog/2018/05/18/algorithm-breakdown-affinity-propagation/
pub struct AffinityPropagation<L> {
    similarity: Array2<Value>,
    responsibility: Array2<Value>,
    availability: Array2<Value>,
    solution: Vec<usize>,
    labels: Vec<L>,
    config: Config,
}

impl<L> AffinityPropagation<L> {
    /// Generate cluster predictions for set of `x` values and `y` labels
    /// - x: 2-D array of (rows=samples, cols=attr_values)
    /// - y: 1-D array of label values attached to each row in `x`
    /// - cfg: Prediction configurations
    pub fn predict<S>(x: Array2<Value>, y: Vec<L>, cfg: Config, s: S)
    where
        S: Similarity,
        L: std::marker::Send + std::fmt::Debug,
    {
        assert_eq!(x.dim().0, y.len(), "`x` n_row != `y` length");
        let mut ap = AffinityPropagation::new(s.similarity(x), y, cfg);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(cfg.workers)
            .build()
            .unwrap();
        pool.scope(move |_| {
            for i in 0..cfg.iterations {
                println!("Beginning iteration {}", i + 1);
                ap.update_r();
                ap.update_a();
                let sol = &ap.availability + &ap.responsibility;
                let mut exemplars: Vec<usize> = Vec::new();
                sol.axis_iter(Axis(1))
                    .into_par_iter()
                    .map(|col| AffinityPropagation::<L>::max_argmax(col).0)
                    .collect_into_vec(&mut exemplars);
                let exemplars: HashSet<usize> = HashSet::from_iter(exemplars.into_iter());
                let solution = Vec::from_iter(exemplars.into_iter());
                if solution == ap.solution {
                    break;
                }
                ap.solution = solution;
                println!(
                    "Iter({}): {:?}",
                    i,
                    ap.solution
                        .iter()
                        .map(|e_idx| ap.labels.get(*e_idx).unwrap())
                        .collect::<Vec<&L>>()
                );
            }
            println!(
                "Final: {:?}",
                ap.solution
                    .iter()
                    .map(|e_idx| ap.labels.get(*e_idx).unwrap())
                    .collect::<Vec<&L>>()
            );
        });
    }

    fn new(x: Array2<Value>, y: Vec<L>, cfg: Config) -> Self {
        let dim = x.dim();
        let y_len = y.len();
        Self {
            similarity: x,
            responsibility: Array2::zeros(dim),
            availability: Array2::zeros(dim),
            config: cfg,
            labels: y,
            solution: vec![0; y_len],
        }
    }

    fn update_r(&mut self) {
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
            .map(|col| AffinityPropagation::<L>::max_argmax(col))
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
            .map(|col| AffinityPropagation::<L>::max_argmax(col))
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
    }

    fn update_a(&mut self) {
        let damping = self.config.damping;
        // a = np.array(R)
        let mut a = self.responsibility.clone();

        // a[a < 0] = 0
        a.par_mapv_inplace(|c| if c < 0. { 0. } else { c });

        // np.fill_diagonal(a, 0)
        a.diag_mut().par_map_inplace(|c| *c = 0.);

        // a = a.sum(axis=0) # columnwise sum
        // a = a + R[k_k_idx, k_k_idx]
        let mut a = a.sum_axis(Axis(0));
        Zip::from(&mut a)
            .and(self.responsibility.diag())
            .par_for_each(|a, &r| *a = *a + r);

        // a = np.ones(A.shape) * a
        let mut a = Array2::<Value>::ones(self.availability.dim()) * a;

        // a -= np.clip(R, 0, np.inf)
        Zip::from(&mut a)
            .and(&self.responsibility)
            .par_for_each(|a, &r| {
                let r = if r < 0. { r } else { 0. };
                *a -= r;
            });

        // a[a > 0] = 0
        a.par_mapv_inplace(|c| if c > 0. { 0. } else { c });

        // w = np.array(R)
        // np.fill_diagonal(w, 0)
        let mut w = self.responsibility.clone();
        w.diag_mut().par_map_inplace(|d| *d = 0.);

        // w[w < 0] = 0
        w.par_mapv_inplace(|c| if c < 0. { 0. } else { c });

        // a[k_k_idx, k_k_idx] = w.sum(axis=0) # column wise sum
        Zip::from(a.diag_mut())
            .and(&w.sum_axis(Axis(0)))
            .par_for_each(|_a, &_w| *_a = _w);

        // A = A * damping + (1 - damping) * a
        Zip::from(&mut self.availability)
            .and(&a)
            .par_for_each(|av, &_a| *av = *av * damping + (1. - damping) * _a);
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
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::affinity_propagation::{AffinityPropagation, Config};
    use ndarray::{arr2, array, Array2};

    #[test]
    fn init() {
        let x: Array2<Value> = arr2(&[[1., 2., 3.], [4., 5., 6.]]);
        let y = vec![0, 1];
        AffinityPropagation::predict(x, y, Config::default(), Euclidean {});
    }
}