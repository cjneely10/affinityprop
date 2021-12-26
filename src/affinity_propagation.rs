use ndarray::{Array1, Array2, ArrayView, Axis, Dim, Zip};
use num_traits::Float;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use std::collections::{HashMap, HashSet};
use std::fmt::Display;

pub trait Similarity<F> {
    type UserType;
    type FloatType;
    fn similarity(self, x: Array2<Self::UserType>) -> Array2<F>;
}

pub struct Euclidean;

impl Default for Euclidean {
    fn default() -> Self {
        Euclidean {}
    }
}

impl<F> Similarity<F> for Euclidean
where
    F: Float,
{
    type UserType = F;
    type FloatType = F;

    /// Row-by-row similarity calculation using negative euclidean distance
    fn similarity(self, x: Array2<Self::UserType>) -> Array2<F> {
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

/// Implementation derived from sklearn AffinityPropagation implementation
/// https://github.com/scikit-learn/scikit-learn/blob/0d378913b/sklearn/cluster/_affinity_propagation.py#L38
pub struct AffinityPropagation<F> {
    similarity: Array2<F>,
    responsibility: Array2<F>,
    availability: Array2<F>,
    pub exemplar_map: HashMap<usize, Vec<usize>>,
    damping: F,
    threads: usize,
    max_iterations: usize,
    convergence_iter: usize,
    preference: F,
    converged: bool,
}

impl<F> AffinityPropagation<F>
where
    F: Float + Send + Sync,
{
    /// Generate cluster predictions for set of `x` values and `y` labels
    /// - x: 2-D array of (rows=samples, cols=attr_values)
    /// - y: 1-D array of label values attached to each row in `x`
    /// - s: Similarity calculator -> must generate an N x N matrix
    pub fn new<S>(
        x: Array2<S::UserType>,
        s: S,
        damping: F,
        threads: usize,
        max_iterations: usize,
        convergence_iter: usize,
        preference: F,
    ) -> Self
    where
        S: Similarity<F>,
    {
        let s = s.similarity(x);
        let s_dim = s.dim();
        assert_eq!(s_dim.0, s_dim.1, "similarity dim must be NxN");
        let mut ap = Self {
            similarity: s,
            responsibility: Array2::<F>::zeros(s_dim),
            availability: Array2::<F>::zeros(s_dim),
            exemplar_map: HashMap::new(),
            damping,
            threads,
            max_iterations,
            convergence_iter,
            preference,
            converged: false,
        };
        ap.add_preference_to_sim();
        ap
    }

    pub fn with_defaults<S>(x: Array2<S::UserType>, s: S) -> Self
    where
        S: Similarity<F>,
    {
        let s = s.similarity(x);
        let s_dim = s.dim();
        assert_eq!(s_dim.0, s_dim.1, "similarity dim must be NxN");
        let mut ap = Self {
            similarity: s,
            responsibility: Array2::<F>::zeros(s_dim),
            availability: Array2::<F>::zeros(s_dim),
            exemplar_map: HashMap::new(),
            damping: F::from(0.5).unwrap(),
            threads: 4,
            max_iterations: 100,
            convergence_iter: 10,
            preference: F::from(-10.).unwrap(),
            converged: false,
        };
        ap.add_preference_to_sim();
        ap
    }

    pub fn predict(&mut self) {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.threads)
            .build()
            .unwrap();
        pool.scope(move |_| {
            for _ in 0..self.convergence_iter {
                self.update_r();
                self.update_a();
            }
            let mut final_exemplars = self.generate_exemplars();
            for _ in self.convergence_iter..self.max_iterations {
                self.update_r();
                self.update_a();
                let sol_map = self.generate_exemplars();
                if final_exemplars.len() == sol_map.len()
                    && final_exemplars.iter().all(|k| sol_map.contains(k))
                {
                    self.converged = true;
                    break;
                }
                final_exemplars = sol_map;
            }
            self.exemplar_map = self.generate_exemplar_map(final_exemplars);
        });
    }

    pub fn results(&self) -> &HashMap<usize, Vec<usize>> {
        &self.exemplar_map
    }

    pub fn display_results<L>(&self, labels: &[L])
    where
        L: Display + Send + Clone + ToString,
    {
        println!(
            "Converged={} nClusters={} nSamples={}",
            self.converged,
            self.exemplar_map.len(),
            self.similarity.dim().0
        );
        self.exemplar_map
            .iter()
            .enumerate()
            .for_each(|(idx, (key, value))| {
                println!(
                    ">Cluster={} size={} exemplar={}",
                    idx + 1,
                    value.len(),
                    labels[*key]
                );
                println!(
                    "{}",
                    value
                        .iter()
                        .map(|v| labels[*v].to_string())
                        .collect::<Vec<String>>()
                        .join(",")
                );
            });
    }

    fn generate_exemplars(&self) -> HashSet<usize> {
        let idx = Array1::<F>::range(
            F::from(0.).unwrap(),
            F::from(self.similarity.dim().0).unwrap(),
            F::from(1.).unwrap(),
        );
        let zero = F::from(0.).unwrap();
        let values: Vec<isize> = Vec::from_iter(
            Zip::from(&self.responsibility.diag())
                .and(&self.availability.diag())
                .and(&idx)
                .par_map_collect(|&r, &a, &i: &F| {
                    if r + a > zero {
                        return i.to_isize().unwrap();
                    }
                    return -1;
                }),
        );
        let values: Vec<usize> = values
            .par_iter()
            .filter(|v| **v >= 0)
            .map(|c| *c as usize)
            .collect();
        HashSet::from_iter(values.into_iter())
    }

    fn generate_exemplar_map(&self, sol_map: HashSet<usize>) -> HashMap<usize, Vec<usize>> {
        let mut exemplar_map = HashMap::from_iter(sol_map.into_iter().map(|x| (x, vec![])));
        let idx = Array1::range(
            F::from(0.).unwrap(),
            F::from(self.similarity.dim().0).unwrap(),
            F::from(1.).unwrap(),
        );
        let max_results = Zip::from(&idx)
            .and(self.similarity.axis_iter(Axis(1)))
            .par_map_collect(|&i, col| {
                let i = i.to_usize().unwrap();
                if exemplar_map.contains_key(&i) {
                    return (i, i);
                }
                // Collect into (idx, value)
                let mut col_data: Vec<(usize, F)> =
                    col.into_iter().map(|v| v.clone()).enumerate().collect();
                // Sort by value
                col_data.sort_by(|&v1, &v2| v2.1.partial_cmp(&v1.1).unwrap());
                // Return highest value that is present in exemplar map keys
                for v in col_data.iter() {
                    if exemplar_map.contains_key(&v.0) {
                        return (v.0, i);
                    }
                }
                return (col_data[0].0, i);
            });
        max_results
            .into_iter()
            .for_each(|max_val| exemplar_map.get_mut(&max_val.0).unwrap().push(max_val.1));
        exemplar_map
    }

    fn add_preference_to_sim(&mut self) {
        let pref = self.preference;
        self.similarity.diag_mut().par_map_inplace(|v| *v = pref);
    }

    fn update_r(&mut self) {
        let mut tmp: Array2<F> = Array2::zeros(self.similarity.dim());
        Zip::from(&mut tmp)
            .and(&self.similarity)
            .and(&self.availability)
            .par_for_each(|t, &s, &a| *t = s + a);

        let combined =
            Zip::from(tmp.axis_iter(Axis(1))).par_map_collect(|col| Self::max_argmax(col));

        let max_idx: Array1<usize> = combined.iter().map(|c| c.0).collect();
        let max1: Array1<F> = combined.iter().map(|c| c.1).collect();

        let neg_inf = F::from(-1.).unwrap() * F::from(f64::INFINITY).unwrap();
        Zip::from(tmp.axis_iter_mut(Axis(1)))
            .and(&max_idx)
            .par_for_each(|mut t, &m| {
                t[m] = neg_inf;
            });

        let max2 = Zip::from(tmp.axis_iter(Axis(1))).par_map_collect(|col| Self::max_argmax(col).1);

        let mut tmp: Array2<F> = Zip::from(&self.similarity)
            .and(
                &max1
                    .insert_axis(Axis(1))
                    .broadcast(self.similarity.dim())
                    .unwrap(),
            )
            .par_map_collect(|&s, &m| s - m);

        Zip::from(tmp.axis_iter_mut(Axis(0)))
            .and(self.similarity.axis_iter(Axis(0)))
            .and(&max_idx)
            .and(&max2)
            .par_for_each(|mut t, s, &m_idx, &m2| t[m_idx] = s[m_idx] - m2);

        let damping = self.damping;
        let inv_damping = F::from(1.).unwrap() - damping;
        tmp.par_map_inplace(|v| *v = *v * inv_damping);
        self.responsibility.par_map_inplace(|v| *v = *v * damping);
        Zip::from(&mut self.responsibility)
            .and(&tmp)
            .par_for_each(|r, &t| *r = *r + t);
    }

    fn update_a(&mut self) {
        let mut tmp = self.responsibility.clone();
        let zero = F::from(0.).unwrap();
        tmp.par_map_inplace(|v| {
            if *v < zero {
                *v = zero;
            }
        });
        Zip::from(tmp.diag_mut())
            .and(self.responsibility.diag())
            .par_for_each(|t, &r| *t = r);

        let mut tmp = Zip::from(&tmp)
            .and(
                &tmp.sum_axis(Axis(0))
                    .insert_axis(Axis(1))
                    .broadcast(tmp.dim())
                    .unwrap(),
            )
            .par_map_collect(|&t, &s| t - s);

        let tmp_diag = tmp.diag().to_owned();
        tmp.par_map_inplace(|v| {
            if *v < zero {
                *v = zero;
            }
        });
        Zip::from(tmp.diag_mut())
            .and(&tmp_diag)
            .par_for_each(|t, d| *t = *d);

        let damping = self.damping;
        let inv_damping = F::from(1.).unwrap() - damping;
        tmp.par_map_inplace(|v| *v = *v * inv_damping);
        self.availability.par_map_inplace(|v| *v = *v * damping);
        Zip::from(&mut self.availability)
            .and(&tmp)
            .par_for_each(|a, &t| *a = *a - t);
    }

    fn max_argmax(data: ArrayView<F, Dim<[usize; 1]>>) -> (usize, F) {
        let mut max_pos = 0;
        let mut max: F = data[0];
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
    fn valid_update_r() {
        todo!()
    }

    fn valid_update_a() {
        todo!()
    }

    fn valid_select_exemplars() {
        todo!()
    }

    fn valid_gather_members() {
        todo!()
    }
}
