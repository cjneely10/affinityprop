use std::collections::{HashMap, HashSet};

use ndarray::{Array1, Array2, ArrayView, Axis, Dim, Zip};
use num_traits::Float;

use crate::{Cluster, ClusterMap, ClusterResults, Idx};

/// House the contents of the algorithm itself inside a droppable object.
///
/// Use this struct to do the actual calculations associated with AP,
/// including holding (large) responsibility and availability arrays.
///
/// Also store pre-calculated constants (like 0, neg inf) to remove unneeded
/// constructions.
pub(crate) struct APAlgorithm<F> {
    similarity: Array2<F>,
    responsibility: Array2<F>,
    availability: Array2<F>,
    tmp: Array2<F>,
    damping: F,
    inv_damping: F,
    neg_inf: F,
    zero: F,
    idx: Array1<F>,
}

impl<F> APAlgorithm<F>
where
    F: Float + Send + Sync,
{
    /// Create algorithm with initial data
    ///
    /// - damping: (0, 1)
    /// - preference: If None, will calculate median similarity
    /// - s: Similarity matrix
    pub(crate) fn new(damping: F, s: Array2<F>) -> Self {
        let s_dim = s.dim();
        let zero = F::from(0.).unwrap();
        Self {
            similarity: s,
            responsibility: Array2::zeros(s_dim),
            availability: Array2::zeros(s_dim),
            tmp: Array2::zeros(s_dim),
            damping,
            inv_damping: F::from(1.).unwrap() - damping,
            neg_inf: F::from(-1.).unwrap() * F::infinity(),
            zero,
            idx: Self::generate_idx(zero, s_dim.0),
        }
    }

    /// Run prediction algorithm
    ///
    /// Repeat responsibility and availability updates for `convergence_iter` rounds, then begin
    /// identifying exemplars. Once the exemplars start changing, or `max_iterations` is reached,
    /// collect and return clusters  
    pub(crate) fn predict(
        &mut self,
        convergence_iter: usize,
        max_iterations: usize,
    ) -> ClusterResults {
        let mut has_converged = false;
        for _ in 0..convergence_iter {
            self.update();
        }
        let mut final_exemplars = self.generate_exemplars();
        for _ in convergence_iter..max_iterations {
            self.update();
            let sol_map = self.generate_exemplars();
            if !sol_map.is_empty()
                && final_exemplars.len() == sol_map.len()
                && final_exemplars.iter().all(|k| sol_map.contains(k))
            {
                has_converged = true;
                break;
            }
            final_exemplars = sol_map;
        }
        (has_converged, self.generate_exemplar_map(final_exemplars))
    }

    /// Update predictions
    fn update(&mut self) {
        self.update_r();
        self.update_a();
    }

    /// Collect currently-predicted exemplars
    ///
    /// Data point is a valid exemplar if the sum of its self-responsibility and
    /// self-availability is positive.
    fn generate_exemplars(&self) -> HashSet<Idx> {
        let values: Vec<Option<Idx>> = Vec::from_iter(
            Zip::from(&self.responsibility.diag())
                .and(&self.availability.diag())
                .and(&self.idx)
                .par_map_collect(|&r, &a, &i: &F| {
                    if r + a > self.zero {
                        return Some(i.to_usize().unwrap());
                    }
                    None
                }),
        );
        HashSet::from_iter(values.into_iter().flatten())
    }

    /// Collect members of each cluster based on their similarity to current exemplars. Exemplar
    /// indices are included in their assigned clusters.
    ///
    /// If no exemplars are currently available, will return an empty map
    fn generate_exemplar_map(&self, sol_map: HashSet<Idx>) -> ClusterMap {
        let mut exemplar_map = HashMap::from_iter(sol_map.into_iter().map(|x| (x, Cluster::new())));
        if exemplar_map.is_empty() {
            return exemplar_map;
        }
        let max_results = Zip::from(&self.idx)
            .and(self.similarity.axis_iter(Axis(1)))
            .par_map_collect(|&i, col| {
                let i = i.to_usize().unwrap();
                if exemplar_map.contains_key(&i) {
                    return (i, i);
                }
                // Collect into (idx, value)
                let mut col_data: Vec<(usize, &F)> = col.into_iter().enumerate().collect();
                // Sort by value
                col_data.sort_by(|&v1, &v2| v2.1.partial_cmp(v1.1).unwrap());
                // Return highest idx that is present in exemplar map
                for v in col_data.iter() {
                    if exemplar_map.contains_key(&v.0) {
                        return (v.0, i);
                    }
                }
                unreachable!()
            });
        max_results
            .into_iter()
            .for_each(|max_val| exemplar_map.get_mut(&max_val.0).unwrap().push(max_val.1));
        exemplar_map
    }

    /// Pre-generate row/col index to reduce number of copies made
    fn generate_idx(zero: F, x_dim: usize) -> Array1<F> {
        Array1::range(zero, F::from(x_dim).unwrap(), F::from(1.).unwrap())
    }

    /// Update responsibilities, follow implementation in sklearn at
    ///
    /// <https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09b/sklearn/cluster/_affinity_propagation.py#L182>
    fn update_r(&mut self) {
        // np.add(A, S, tmp)
        Zip::from(&mut self.tmp)
            .and(&self.similarity)
            .and(&self.availability)
            .par_for_each(|t, &s, &a| *t = s + a);

        let combined =
            Zip::from(self.tmp.axis_iter(Axis(1))).par_map_collect(|col| Self::max_argmax(col));

        // I = np.argmax(tmp, axis=1)
        let max_idx: Array1<usize> = combined.iter().map(|c| c.0).collect();
        // Y = tmp[ind, I]
        let max1: Array1<F> = combined.iter().map(|c| c.1).collect();

        // tmp[ind, I] = -np.inf
        Zip::from(self.tmp.axis_iter_mut(Axis(1)))
            .and(&max_idx)
            .par_for_each(|mut t, &m| {
                t[m] = self.neg_inf;
            });

        // Y2 = np.max(tmp, axis=1)
        let max2 =
            Zip::from(self.tmp.axis_iter(Axis(1))).par_map_collect(|col| Self::max_argmax(col).1);

        // np.subtract(S, Y[:, None], tmp)
        Zip::from(&mut self.tmp)
            .and(
                &Zip::from(&self.similarity)
                    .and(
                        &max1
                            .insert_axis(Axis(1))
                            .broadcast(self.similarity.dim())
                            .unwrap(),
                    )
                    .par_map_collect(|&s, &m| s - m),
            )
            .par_for_each(|t, s| *t = *s);

        // tmp[ind, I] = S[ind, I] - Y2
        Zip::from(self.tmp.axis_iter_mut(Axis(0)))
            .and(self.similarity.axis_iter(Axis(0)))
            .and(&max_idx)
            .and(&max2)
            .par_for_each(|mut t, s, &m_idx, &m2| t[m_idx] = s[m_idx] - m2);

        // tmp *= 1 - damping
        self.tmp.par_map_inplace(|v| *v = *v * self.inv_damping);
        // R *= damping
        self.responsibility
            .par_map_inplace(|v| *v = *v * self.damping);
        // R += tmp
        Zip::from(&mut self.responsibility)
            .and(&self.tmp)
            .par_for_each(|r, &t| *r = *r + t);
    }

    /// Update availabilities, follow implementation in sklearn at
    ///
    /// <https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09b/sklearn/cluster/_affinity_propagation.py#L198>
    fn update_a(&mut self) {
        // np.maximum(R, 0, tmp)
        Zip::from(&mut self.tmp)
            .and(&self.responsibility)
            .par_for_each(|t, r| *t = *r);
        self.tmp.par_map_inplace(|v| {
            if *v < self.zero {
                *v = self.zero;
            }
        });
        // tmp.flat[:: n_samples + 1] = R.flat[:: n_samples + 1]
        Zip::from(&mut self.tmp.diag_mut())
            .and(self.responsibility.diag())
            .par_for_each(|t, &r| *t = r);

        // tmp -= np.sum(tmp, axis=0)
        let sum = self.tmp.sum_axis(Axis(0));
        Zip::from(self.tmp.axis_iter_mut(Axis(0)))
            .and(&sum)
            .par_for_each(|mut t, &s| t.par_map_inplace(|t| *t = *t - s));

        // dA = np.diag(tmp).copy()
        let tmp_diag = self.tmp.diag().to_owned();
        // tmp.clip(0, np.inf, tmp)
        self.tmp.par_map_inplace(|v| {
            if *v < self.zero {
                *v = self.zero;
            }
        });
        // tmp.flat[:: n_samples + 1] = dA
        Zip::from(self.tmp.diag_mut())
            .and(&tmp_diag)
            .par_for_each(|t, d| *t = *d);

        // tmp *= 1 - damping
        self.tmp.par_map_inplace(|v| *v = *v * self.inv_damping);
        // A *= damping
        self.availability
            .par_map_inplace(|v| *v = *v * self.damping);
        // A -= tmp
        Zip::from(&mut self.availability)
            .and(&self.tmp)
            .par_for_each(|a, &t| *a = *a - t);
    }

    /// Collect maximum value and its index from view of data
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
    use std::collections::{HashMap, HashSet};

    use ndarray::{arr2, Array2};
    use rayon::ThreadPool;

    use crate::algorithm::APAlgorithm;
    use crate::preference::place_preference;
    use crate::Preference::Value;

    fn pool(t: usize) -> ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .unwrap()
    }

    fn test_data() -> Array2<f32> {
        arr2(&[
            [0., -7., -6., -12., -17.],
            [-7., 0., -17., -17., -22.],
            [-6., -17., 0., -18., -21.],
            [-12., -17., -18., 0., -3.],
            [-17., -22., -21., -3., 0.],
        ])
    }

    #[test]
    fn valid_select_exemplars() {
        pool(2).scope(move |_| {
            let mut sim = test_data();
            place_preference(&mut sim, Value(-22.));
            let mut calc: APAlgorithm<f32> = APAlgorithm::new(0., sim);
            calc.update();
            let exemplars = calc.generate_exemplars();
            let actual: HashSet<usize> = HashSet::from([0]);
            assert!(
                actual.len() == exemplars.len() && actual.iter().all(|v| exemplars.contains(v))
            );
        });
    }

    #[test]
    fn valid_gather_members() {
        pool(2).scope(move |_| {
            let mut sim = test_data();
            place_preference(&mut sim, Value(-22.));
            let mut calc: APAlgorithm<f32> = APAlgorithm::new(0., sim);
            calc.update();
            let exemplars = calc.generate_exemplar_map(calc.generate_exemplars());
            let actual: HashMap<usize, Vec<usize>> = HashMap::from([(0, vec![0, 1, 2, 3, 4])]);
            assert!(
                actual.len() == exemplars.len()
                    && actual.iter().all(|(idx, values)| {
                        if !exemplars.contains_key(idx) {
                            return false;
                        }
                        let v: HashSet<usize> =
                            HashSet::from_iter(values.iter().map(|v| v.clone()));
                        let a: HashSet<usize> = HashSet::from_iter(
                            exemplars.get(idx).unwrap().iter().map(|v| v.clone()),
                        );
                        return v.len() == a.len() && v.iter().all(|p| v.contains(p));
                    })
            );
        });
    }
}
