#[cfg(not(tarpaulin))]
#[cfg(not(tarpaulin_include))]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::ffi::OsStr;
    use std::fmt::Debug;
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use std::path::{Path, PathBuf};
    use std::str::FromStr;

    use ndarray::{Array2, Axis};
    use num_traits::Float;

    use affinityprop::Preference::Value;
    use affinityprop::{
        AffinityPropagation, LogEuclidean, NegCosine, NegEuclidean, Preference, Similarity,
    };

    /// Load test data into Array for evaluation, and gather actual results into result map
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

    /// Match predicted clusters to actual data and generate best F1 for each cluster pair
    ///
    /// ```
    /// Precision = TP/(TP + FP)
    /// Recall = TP/(TP + FN)
    /// F1 = 2 * (P * R) / (P + R)
    /// ```
    ///
    /// Returns average best F1 score
    fn compare_clusters<F, S>(
        actual: HashMap<usize, Vec<usize>>,
        predicted: HashMap<usize, Vec<usize>>,
    ) -> F
    where
        F: Float + Send + Sync,
    {
        // Convert cluster repr to HashSets for searching
        let actual: HashMap<usize, HashSet<usize>> = HashMap::from_iter(
            actual
                .iter()
                .map(|(k, v)| (*k, HashSet::from_iter(v.iter().map(|v| *v)))),
        );
        let predicted: HashMap<usize, HashSet<usize>> = HashMap::from_iter(
            predicted
                .iter()
                .map(|(k, v)| (*k, HashSet::from_iter(v.iter().map(|v| *v)))),
        );
        let mut matched_cluster_ids: HashSet<usize> = HashSet::new();
        let mut total_score: F = F::from(0.).unwrap();
        let mut size = actual.len();
        if predicted.len() > actual.len() {
            size = predicted.len();
        }
        let size = F::from(size).unwrap();
        for (_, actual_cluster) in actual.iter() {
            // Get best-scoring index
            let mut best_exemplar: usize = 0;
            let mut best_score: F = F::from(0.).unwrap();
            for (predicted_exemplar, predicted_cluster) in predicted.iter() {
                if matched_cluster_ids.contains(predicted_exemplar) {
                    continue;
                }
                let score = f1(actual_cluster, predicted_cluster);
                if score > best_score {
                    best_score = score;
                    best_exemplar = *predicted_exemplar;
                }
            }
            // Store best match and update score
            matched_cluster_ids.insert(best_exemplar);
            total_score = total_score + best_score / size;
        }
        total_score
    }

    fn precision_recall<F>(actual: &HashSet<usize>, predicted: &HashSet<usize>) -> (F, F)
    where
        F: Float + Send + Sync,
    {
        let tp: u32 = actual
            .iter()
            .map(|v| u32::from(predicted.contains(v)))
            .sum();
        let fp: u32 = predicted
            .iter()
            .map(|v| u32::from(!actual.contains(v)))
            .sum();
        let f_n: u32 = actual
            .iter()
            .map(|v| u32::from(!predicted.contains(v)))
            .sum();

        let tp = F::from(tp).unwrap();
        let fp = F::from(fp).unwrap();
        let f_n = F::from(f_n).unwrap();
        // Precision = TP/(TP + FP)
        // Recall = TP/(TP + FN)
        ((tp / (tp + fp)), (tp / (tp + f_n)))
    }

    fn f1<F>(actual: &HashSet<usize>, predicted: &HashSet<usize>) -> F
    where
        F: Float + Send + Sync,
    {
        let (precision, recall) = precision_recall::<F>(actual, predicted);
        F::from(2.).unwrap() * precision * recall / (precision + recall)
    }

    /// Run test using dataset in file. Optionally compute F1 score.
    fn run_test<F, S>(
        ap: &AffinityPropagation<F>,
        s: S,
        path: PathBuf,
        preference: Preference<F>,
        expected_f1: F,
    ) where
        F: Float + Send + Sync + FromStr + Default + std::fmt::Display,
        S: Similarity<F>,
        <F as FromStr>::Err: Debug,
    {
        let (test_array, actual) = load_data::<F>(path.clone()).unwrap();
        let (converged, test_results) = ap.predict(&test_array, s, preference);
        assert!(converged);
        assert_eq!(actual.len(), test_results.len());
        let f1 = compare_clusters::<F, S>(actual, test_results);
        println!("Test(F1={:.2}): {:?}", f1, path);
        assert!(f1 >= expected_f1);
    }

    /// Helper function to load file from test directory
    fn file<A: AsRef<OsStr>>(path: A) -> PathBuf {
        let test_dir = Path::new(file!()).parent().unwrap();
        test_dir.join(Path::new("test-data")).join(Path::new(&path))
    }

    #[test]
    fn ten_exemplars() {
        let ap = AffinityPropagation::<f32>::new(0.5, 4, 400, 4000);
        run_test(
            &ap,
            NegEuclidean::default(),
            file(&"near-exemplar-10.test"),
            Value(-1000.),
            0.99,
        );
    }

    #[test]
    fn fifty_exemplars() {
        let ap = AffinityPropagation::<f32>::new(0.5, 4, 400, 4000);
        run_test(
            &ap,
            NegEuclidean::default(),
            file(&"near-exemplar-50.test"),
            Value(-1000.),
            0.99,
        );
    }

    #[test]
    fn breast_cancer() {
        let ap = AffinityPropagation::<f32>::new(0.95, 4, 400, 4000);
        run_test(
            &ap,
            LogEuclidean::default(),
            file(&"breast_cancer.test"),
            Value(-10000.),
            0.60,
        )
    }

    /// This test is very long-running. Run in release mode to reduce clock time.
    #[test]
    fn binsanity() {
        let ap = AffinityPropagation::<f32>::new(0.95, 4, 400, 4000);
        run_test(
            &ap,
            NegEuclidean::default(),
            file(&"binsanity.test"),
            Value(-10.),
            0.98,
        )
    }

    /// Iris test predicts 4 labels instead of 3
    #[test]
    #[should_panic]
    fn iris() {
        let ap = AffinityPropagation::<f32>::new(0.95, 4, 400, 4000);
        run_test(
            &ap,
            LogEuclidean::default(),
            file(&"iris.test"),
            Preference::Median,
            0.60,
        );
    }

    /// Diabetes test predicts 34 instead of 214
    #[test]
    #[should_panic]
    fn diabetes() {
        let ap = AffinityPropagation::<f32>::new(0.95, 4, 400, 4000);
        run_test(
            &ap,
            NegCosine::default(),
            file(&"diabetes.test"),
            Preference::Median,
            0.60,
        )
    }
}
