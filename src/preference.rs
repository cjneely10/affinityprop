use ndarray::{Array1, Array2, Zip};
use num_traits::Float;

/// Preference is the value representing the degree to which a data point will act as its own exemplar,
/// with lower (more negative) values yielding fewer clusters.
///
/// - Median: Use median similarity value as preference
/// - List: Use provided preference list
/// - Value: Assign all members the same preference value
#[derive(Debug, Clone)]
pub enum Preference<'a, F>
where
    F: Float + Send + Sync,
{
    /// Use the median pairwise similarity as the preference
    Median,
    /// Use a list of preferences, one per input
    List(&'a Array1<F>),
    /// Use a single value as the preference for all inputs
    Value(F),
}

pub(crate) fn place_preference<F>(s: &mut Array2<F>, p: Preference<F>)
where
    F: Float + Send + Sync,
{
    let s_dim = s.dim();
    let preference = match p {
        Preference::Value(pref) => pref,
        Preference::Median => median(s),
        Preference::List(l) => {
            assert!(
                s_dim.0 == l.len(),
                "Preference list length does not match input length!"
            );
            Zip::from(l)
                .and(s.diag_mut())
                .par_for_each(|pref, s_pos| *s_pos = *pref);
            return;
        }
    };
    s.diag_mut().par_map_inplace(|v| *v = preference);
}

/// Computed simply - collect values into vector, sort, and return value at len() / 2
fn median<F>(x: &Array2<F>) -> F
where
    F: Float + Send + Sync,
{
    let mut sorted_values = Vec::new();
    let x_dim_0 = x.dim().0 as usize;
    for i in 0..x_dim_0 {
        for j in (i + 1)..x_dim_0 {
            sorted_values.push(x[[i, j]]);
        }
    }
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted_values[sorted_values.len() / 2]
}

#[cfg(test)]
mod test {
    use ndarray::{arr1, arr2, Array2};
    use rayon::ThreadPool;

    use crate::preference::{median, place_preference};
    use crate::Preference::List;

    fn pool(t: usize) -> ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .unwrap()
    }

    fn test_data() -> Array2<f32> {
        arr2(&[
            [0., -5., -6., -12., -17.],
            [-5., 0., -17., -17., -22.],
            [-6., -17., 0., -18., -21.],
            [-12., -17., -18., 0., -3.],
            [-17., -22., -21., -3., 0.],
        ])
    }

    #[test]
    fn valid_median() {
        assert_eq!(-17., median(&test_data()));
    }

    #[test]
    fn provided_preference_list() {
        pool(2).scope(move |_| {
            let mut sim = test_data();
            let pref_list = arr1(&[-1., -2., -3., -4., -5.]);
            place_preference(&mut sim, List(&pref_list));
        });
    }

    #[test]
    #[should_panic]
    fn invalid_preference_list() {
        pool(2).scope(move |_| {
            let mut sim = test_data();
            let pref_list = arr1(&[-1., -2., -3.]);
            place_preference(&mut sim, List(&pref_list));
        });
    }
}
