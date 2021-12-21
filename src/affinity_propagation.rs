use ndarray::{Array, Array1, Array2, ArrayView, Axis, Dim, Zip};
use std::collections::{HashMap, HashSet};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

pub type Value = f32;

const NEG_INF: Value = (-1. as Value) * Value::INFINITY;

#[derive(Copy, Clone, Debug)]
pub struct Config {
    pub damping: f32,
    pub threads: usize,
    pub max_iterations: usize,
    pub convergence_iter: usize,
    pub preference: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            damping: 0.5,
            preference: -10.,
            threads: 4,
            max_iterations: 1000,
            convergence_iter: 100,
        }
    }
}

pub trait Similarity {
    fn similarity(self, x: Array2<Value>) -> Array2<Value>;
}

pub struct Euclidean;

impl Default for Euclidean {
    fn default() -> Self {
        Euclidean {}
    }
}

impl Similarity for Euclidean {
    fn similarity(self, x: Array2<Value>) -> Array2<Value> {
        let x_dim = x.dim();
        let mut out = Array2::<Value>::zeros((x_dim.0, x_dim.0));
        x.axis_iter(Axis(0)).enumerate().for_each(|(idx1, row1)| {
            x.axis_iter(Axis(0)).enumerate().for_each(|(idx2, row2)| {
                let mut row_diff = &row1 - &row2;
                row_diff.mapv_inplace(|a| a.powi(2));
                out[[idx1, idx2]] = -1. * row_diff.sum();
            });
        });
        out
    }
}

/// Implementation derived from:
/// https://www.ritchievink.com/blog/2018/05/18/algorithm-breakdown-affinity-propagation/
pub struct AffinityPropagation {
    similarity: Array2<Value>,
    responsibility: Array2<Value>,
    availability: Array2<Value>,
    labels: Vec<String>,
    config: Config,
}

impl AffinityPropagation {
    /// Generate cluster predictions for set of `x` values and `y` labels
    /// - x: 2-D array of (rows=samples, cols=attr_values)
    /// - y: 1-D array of label values attached to each row in `x`
    /// - cfg: Prediction configurations
    pub fn predict<S>(x: Array2<Value>, y: Vec<String>, cfg: Config, s: S)
    where
        S: Similarity + std::marker::Send,
    {
        let x_dim = x.dim();
        assert_eq!(x_dim.0, y.len(), "`x` n_row != `y` length");
        let mut ap = Self::new(s.similarity(x), y, cfg);
        let mut conv_iterations = 0;
        let mut final_sol = Array2::zeros(ap.availability.dim());
        let mut final_exemplars = HashSet::new();
        println!("Beginning clustering...");
        // println!("{:?}", ap.similarity);
        for i in 0..cfg.max_iterations {
            ap.update_r();
            ap.update_a();
            let sol = &ap.availability + &ap.responsibility;
            if final_sol.abs_diff_eq(&sol, 1e-8) {
                break;
            } else {
                let sol_map = Self::generate_exemplars(&sol);
                if final_exemplars.len() == sol_map.len()
                    && final_exemplars.iter().all(|k| sol_map.contains(k))
                {
                    conv_iterations += 1;
                    if conv_iterations == ap.config.convergence_iter {
                        break;
                    }
                } else {
                    conv_iterations = 0;
                }
                final_exemplars = sol_map;
                final_sol = sol;
            }
            if (i + 1) % 100 == 0 {
                println!("Iter({}) nClusters: {}", i + 1, final_exemplars.len());
            }
        }
        let exemplars = Self::generate_exemplar_map(&final_sol)
            .keys()
            .map(|v| ap.labels.get(*v).unwrap())
            .collect::<Vec<&String>>();
        println!("nClusters: {}, nSamples: {}", exemplars.len(), x_dim.0);
    }

    fn generate_exemplars(sol: &Array2<Value>) -> HashSet<usize> {
        let mut exemplars = HashSet::new();
        sol.axis_iter(Axis(1)).for_each(|col| {
            let exemplar = Self::max_argmax(col);
            exemplars.insert(exemplar.0);
        });
        exemplars
    }

    fn generate_exemplar_map(sol: &Array2<Value>) -> HashMap<usize, Vec<usize>> {
        let mut exemplar_map = HashMap::new();
        sol.axis_iter(Axis(1)).enumerate().for_each(|(idx, col)| {
            let exemplar = Self::max_argmax(col);
            let exemplar_label = exemplar.0;
            if !exemplar_map.contains_key(&exemplar_label) {
                exemplar_map.insert(exemplar_label.to_owned(), vec![]);
            }
            exemplar_map.get_mut(&exemplar_label).unwrap().push(idx);
        });
        exemplar_map
    }

    fn new(x: Array2<Value>, y: Vec<String>, cfg: Config) -> Self {
        let x_dim_0 = x.dim();
        let mut ap = Self {
            similarity: x,
            responsibility: Array2::zeros(x_dim_0),
            availability: Array2::zeros(x_dim_0),
            config: cfg,
            labels: y,
        };
        ap.add_preference_to_sim();
        ap
    }

    fn add_preference_to_sim(&mut self) {
        let pref = self.config.preference;
        self.similarity.diag_mut().map_inplace(|v| *v = pref);
        let dim = self.similarity.dim().0;
        self.similarity = &self.similarity * Array::random((dim, dim), Uniform::new(0., 1.));
    }

    fn update_r(&mut self) {
        let mut tmp: Array2<Value> = &self.availability + &self.similarity;

        let mut max_idx = Vec::new();
        let mut max1 = Vec::new();
        tmp.axis_iter(Axis(1)).for_each(|col| {
            let max = Self::max_argmax(col);
            max_idx.push(max.0);
            max1.push(max.1);
        });

        let max_idx: Array1<usize> = max_idx.into();
        let max1: Array1<Value> = max1.into();

        Zip::from(tmp.axis_iter_mut(Axis(1)))
            .and(&max_idx)
            .for_each(|mut t, &m| t[m] = NEG_INF);

        let mut max2 = Vec::new();
        tmp.axis_iter(Axis(1)).for_each(|col| {
            let max = Self::max_argmax(col);
            max2.push(max.1);
        });
        let max2: Array1<Value> = max2.into();

        let mut tmp: Array2<Value> = &self.similarity - max1.insert_axis(Axis(1));
        Zip::from(tmp.axis_iter_mut(Axis(0)))
            .and(self.similarity.axis_iter(Axis(0)))
            .and(&max_idx)
            .and(&max2)
            .for_each(|mut t, s, &m_idx, &m2| t[m_idx] = s[m_idx] - m2);

        let damping = self.config.damping;
        let inv_damping = 1. - damping;
        tmp.mapv_inplace(|v| v * inv_damping);
        self.responsibility.mapv_inplace(|v| v * damping);
        self.responsibility = &self.responsibility + tmp;
    }

    fn update_a(&mut self) {
        let mut tmp = self.responsibility.clone();
        tmp.mapv_inplace(|v| if v < 0. { 0. } else { v });
        Zip::from(tmp.diag_mut())
            .and(self.responsibility.diag())
            .for_each(|t, &r| *t = r);

        tmp = &tmp - tmp.sum_axis(Axis(0)).insert_axis(Axis(1));
        let tmp_diag = tmp.diag().to_owned();
        tmp.mapv_inplace(|v| if v < 0. { 0. } else { v });
        Zip::from(tmp.diag_mut())
            .and(&tmp_diag)
            .for_each(|t, d| *t = *d);

        let damping = self.config.damping;
        let inv_damping = 1. - damping;
        tmp.mapv_inplace(|v| v * inv_damping);
        self.availability.mapv_inplace(|v| v * damping);
        self.availability = &self.availability - tmp;
    }

    fn max_argmax(data: ArrayView<Value, Dim<[usize; 1]>>) -> (usize, Value) {
        let mut max_pos = 0;
        let mut max: Value = data[0];
        data.iter().enumerate().for_each(|(idx, val)| {
            if *val > max {
                max = *val;
                max_pos = idx;
            }
        });
        (max_pos, max)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::affinity_propagation::{AffinityPropagation, Config};
    use ndarray::{arr2, Array2};

    #[test]
    fn init() {
        let x: Array2<Value> = arr2(&[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]);
        let y = vec!["1".to_string(), "2".to_string(), "3".to_string()];
        AffinityPropagation::predict(x, y, Config::default(), Euclidean::default());
    }
}
