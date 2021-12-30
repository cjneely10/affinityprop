use std::collections::{HashMap, HashSet};

use ndarray::{Array1, Array2, ArrayView, Axis, Dim, Zip};
use num_traits::Float;

pub(crate) struct APAlgorithm<F> {
    similarity: Array2<F>,
    responsibility: Array2<F>,
    availability: Array2<F>,
    damping: F,
    neg_inf: F,
}

impl<F> APAlgorithm<F>
where
    F: Float + Send + Sync,
{
    pub(crate) fn new(damping: F, preference: F, s: Array2<F>) -> Self {
        let s_dim = s.dim();
        let mut calculation = Self {
            similarity: s,
            responsibility: Array2::zeros(s_dim),
            availability: Array2::zeros(s_dim),
            damping,
            neg_inf: F::from(-1.).unwrap() * F::infinity(),
        };
        calculation
            .similarity
            .diag_mut()
            .par_map_inplace(|v| *v = preference);
        calculation
    }

    pub(crate) fn update(&mut self) {
        self.update_r();
        self.update_a();
    }

    pub(crate) fn generate_exemplars(&self) -> HashSet<usize> {
        let idx = self.generate_idx();
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
        HashSet::from_iter(values.into_iter().filter(|v| *v >= 0).map(|c| c as usize))
    }

    pub(crate) fn generate_exemplar_map(
        &self,
        sol_map: HashSet<usize>,
    ) -> HashMap<usize, Vec<usize>> {
        let mut exemplar_map = HashMap::from_iter(sol_map.into_iter().map(|x| (x, vec![])));
        let idx = self.generate_idx();
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
                unreachable!()
            });
        max_results
            .into_iter()
            .for_each(|max_val| exemplar_map.get_mut(&max_val.0).unwrap().push(max_val.1));
        exemplar_map
    }

    fn generate_idx(&self) -> Array1<F> {
        Array1::range(
            F::from(0.).unwrap(),
            F::from(self.similarity.dim().0).unwrap(),
            F::from(1.).unwrap(),
        )
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

        Zip::from(tmp.axis_iter_mut(Axis(1)))
            .and(&max_idx)
            .par_for_each(|mut t, &m| {
                t[m] = self.neg_inf;
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
    use crate::algorithm::APAlgorithm;
    use ndarray::{arr2, Array2};
    use rayon::ThreadPool;
    use std::collections::{HashMap, HashSet};

    fn pool(t: usize) -> ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .unwrap()
    }

    fn test_data() -> Array2<f32> {
        arr2(&[
            [-0., -7., -6., -12., -17.],
            [-7., -0., -17., -17., -22.],
            [-6., -17., -0., -18., -21.],
            [-12., -17., -18., -0., -3.],
            [-17., -22., -21., -3., -0.],
        ])
    }

    #[test]
    fn valid_select_exemplars() {
        pool(2).scope(move |_| {
            let sim = test_data();
            let mut calc: APAlgorithm<f32> = APAlgorithm::new(0., -22., sim);
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
            let sim = test_data();
            let mut calc: APAlgorithm<f32> = APAlgorithm::new(0., -22., sim);
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
