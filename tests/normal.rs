use std::collections::HashMap;
use std::ffi::OsStr;
use std::fmt::Debug;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use ndarray::{Array2, Axis};
use num_traits::Float;

use affinityprop::{AffinityPropagation, NegEuclidean};

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

fn run_test(path: PathBuf) {
    let (test_array, test_results) = load_data::<f32>(path).unwrap();
    let ap = AffinityPropagation::new(-1000., 0.5, 4, 400, 4000);
    let (converged, results) = ap.predict(&test_array, NegEuclidean::default());
    assert!(converged);
    assert_eq!(test_results.len(), results.len());
}

fn file<A: AsRef<OsStr>>(path: A) -> PathBuf {
    let test_dir = Path::new(file!()).parent().unwrap();
    test_dir.join(Path::new("data")).join(Path::new(&path))
}

#[test]
fn ten_exemplars() {
    run_test(file(&"near-exemplar-10.test"));
}

#[test]
fn one_hundred_exemplars() {
    run_test(file(&"near-exemplar-100.test"));
}
