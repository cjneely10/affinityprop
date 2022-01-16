use std::collections::HashMap;
use std::ffi::OsStr;
use std::fmt::Debug;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use ndarray::{Array2, ArrayView1, Axis};
use num_traits::Float;

use affinityprop::{AffinityPropagation, NegEuclidean, Similarity};

fn load_data<F>(test_file: PathBuf) -> std::io::Result<(Array2<F>, HashMap<usize, Vec<usize>>)>
where
    F: Float + Send + Sync + FromStr + Default,
    <F as FromStr>::Err: Debug,
{
    let reader = BufReader::new(File::open(test_file)?);
    let mut test_data_map: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut test_data = Vec::new();
    reader
        .lines()
        .map(|l| l.unwrap())
        .enumerate()
        .for_each(|(idx, line)| {
            let mut line = line.split(" ");
            let id = line.next().unwrap().parse::<usize>().unwrap();
            if !test_data_map.contains_key(&id) {
                test_data_map.insert(id, vec![]);
            }
            test_data_map.get_mut(&id).unwrap().push(idx);
            let point: Vec<F> = line.map(|c| c.parse::<F>().unwrap()).collect();
            test_data.push(point);
        });
    let mut out = Array2::<F>::default((test_data.len(), test_data[0].len()));
    out.axis_iter_mut(Axis(0))
        .enumerate()
        .for_each(|(idx1, mut row)| {
            row.iter_mut().enumerate().for_each(|(idx2, col)| {
                *col = test_data[idx1][idx2];
            });
        });
    Ok((out, test_data_map))
}

#[derive(Default, Clone)]
struct AbsLogEuclidean;

impl<F> Similarity<F> for AbsLogEuclidean
where
    F: Float + Send + Sync,
{
    fn similarity(&self, a: &ArrayView1<F>, b: &ArrayView1<F>) -> F {
        let mut row_diff = a - b;
        row_diff.map_inplace(|_a| *_a = (*_a).powi(2).log2());
        row_diff.sum().abs()
    }
}

fn run_test<F, S>(ap: &AffinityPropagation<F>, s: S, path: PathBuf)
where
    F: Float + Send + Sync + FromStr + Default,
    S: Similarity<F>,
    <F as FromStr>::Err: Debug,
{
    let (test_array, actual) = load_data::<F>(path).unwrap();
    let (converged, test_results) = ap.predict(&test_array, s);
    assert!(converged);
    assert_eq!(actual.len(), test_results.len());
}

fn file<A: AsRef<OsStr>>(path: A) -> PathBuf {
    let test_dir = Path::new(file!()).parent().unwrap();
    test_dir.join(Path::new("test-data")).join(Path::new(&path))
}

#[test]
fn ten_exemplars() {
    let ap = AffinityPropagation::<f32>::new(Some(-1000.), 0.5, 4, 400, 4000);
    run_test(&ap, NegEuclidean::default(), file(&"near-exemplar-10.test"));
}

#[test]
fn fifty_exemplars() {
    let ap = AffinityPropagation::<f32>::new(Some(-1000.), 0.5, 4, 400, 4000);
    run_test(&ap, NegEuclidean::default(), file(&"near-exemplar-50.test"));
}

// #[test]
// fn iris() {
//     let ap = AffinityPropagation::<f32>::new(None, 0.95, 4, 400, 4000);
//     run_test(&ap, NegEuclidean::default(), file(&"iris.test"));
// }

#[test]
fn breast_cancer() {
    let ap = AffinityPropagation::<f32>::new(Some(-10000.), 0.95, 4, 400, 4000);
    run_test(&ap, AbsLogEuclidean::default(), file(&"breast_cancer.test"))
}

// #[test]
// fn diabetes() {
//     let ap = AffinityPropagation::<f32>::new(None, 0.5, 4, 400, 4000);
//     run_test(&ap, NegCosine::default(), file(&"diabetes.test"))
// }

// #[test]
// fn digits() {
//     let ap = AffinityPropagation::<f32>::new(-1000., 0.95, 4, 400, 4000);
//     run_test(&ap, NegEuclidean::default(), file(&"digits.test"))
// }
