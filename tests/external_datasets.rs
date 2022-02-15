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
    use num_traits::{abs, Float};

    use affinityprop::Preference::Value;
    use affinityprop::{
        AffinityPropagation, ClusterMap, Idx, LogEuclidean, NegCosine, NegEuclidean, Preference,
        Similarity,
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

    fn choose_2<F: Float + Send + Sync>(n: usize) -> F {
        let f;
        match n {
            n if n < 2 => f = F::from(0.).unwrap(),
            n if n == 2 => f = F::from(1.).unwrap(),
            _ => {
                let mut total = 1;
                for i in n - 2 + 1..=n {
                    total *= i;
                }
                f = F::from(total).unwrap() / F::from(2.).unwrap();
            }
        }
        f
    }

    /// Test data is hand-written
    #[test]
    fn choose_2_is_valid() {
        assert_eq!(choose_2::<f32>(0), 0.);
        assert_eq!(choose_2::<f32>(1), 0.);
        assert_eq!(choose_2::<f32>(2), 1.);
        assert_eq!(choose_2::<f32>(3), 3.);
        assert_eq!(choose_2::<f32>(4), 6.);
    }

    fn adj_rand_score<F: Float + Send + Sync>(labels_true: &[Idx], labels_pred: &[Idx]) -> F {
        let zero = F::from(0.).unwrap();
        let half = F::from(0.5).unwrap();
        // Pairwise counting of clustering labels
        let mut comparison_matrix: Array2<usize> =
            Array2::zeros((labels_true.len(), labels_pred.len()));
        labels_true
            .iter()
            .zip(labels_pred.iter())
            .for_each(|(x, y)| {
                comparison_matrix[[*x, *y]] += 1;
            });

        let comb: F = comparison_matrix.map(|v| choose_2(*v)).sum();
        let row: F = comparison_matrix
            .axis_iter(Axis(0))
            .fold(zero, |acc, row| acc + choose_2(row.sum()));
        let col: F = comparison_matrix
            .axis_iter(Axis(1))
            .fold(zero, |acc, col| acc + choose_2(col.sum()));
        (comb - ((row * col) / choose_2(labels_true.len())))
            / (half * (row + col) - ((row * col) / choose_2(labels_true.len())))
    }

    /// Test data derived from
    /// <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html>
    #[test]
    fn ari_is_valid() {
        let a = vec![0, 0, 1, 1];
        let b = vec![1, 1, 0, 0];
        let c = vec![0, 0, 1, 2];
        let d = vec![0, 1, 2, 3];
        let e = vec![0; 4];
        assert!(abs(adj_rand_score::<f32>(&a, &a) - 1.0) <= 0.1);
        assert!(abs(adj_rand_score::<f32>(&a, &b) - 1.0) <= 0.1);
        assert!(abs(adj_rand_score::<f32>(&c, &a) - 0.57) <= 0.01);
        assert!(abs(adj_rand_score::<f32>(&a, &c) - 0.57) <= 0.01);
        assert!(abs(adj_rand_score::<f32>(&e, &d) - 0.0) <= 0.1);
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

    fn cluster_map_to_vector(map: &ClusterMap) -> Vec<Idx> {
        let mut out = vec![0; map.iter().fold(0, |acc, val| acc + val.1.len())];
        map.iter().for_each(|(cluster_label, cluster)| {
            cluster
                .into_iter()
                .for_each(|point_label| out[*point_label] = *cluster_label)
        });
        out
    }

    fn generate_label_lists(actual: &ClusterMap, predicted: &ClusterMap) -> (Vec<Idx>, Vec<Idx>) {
        (
            cluster_map_to_vector(actual),
            cluster_map_to_vector(predicted),
        )
    }

    /// Run test using dataset in file. Optionally compute F1 score.
    fn run_test<F, S>(
        ap: &AffinityPropagation<F>,
        s: S,
        path: PathBuf,
        preference: Preference<F>,
        expected_f1: F,
        expected_ari: F,
    ) where
        F: Float + Send + Sync + FromStr + Default + std::fmt::Display,
        S: Similarity<F>,
        <F as FromStr>::Err: Debug,
    {
        let (test_array, actual) = load_data::<F>(path.clone()).unwrap();
        let (converged, test_results) = ap.predict(&test_array, s, preference);
        assert!(converged);
        assert_eq!(actual.len(), test_results.len());
        let (actual_labels, predicted_labels) = generate_label_lists(&actual, &test_results);
        let ari = adj_rand_score::<F>(&actual_labels, &predicted_labels);
        let f1 = compare_clusters::<F, S>(actual, test_results);
        println!("Test( F1={:.3}): {:?}", f1, path);
        println!("Test(ARI={:.3}): {:?}", ari, path);
        assert!(f1 >= expected_f1);
        assert!(ari >= expected_ari);
    }

    /// Helper function to load file from test directory
    fn file<A: AsRef<OsStr>>(path: A) -> PathBuf {
        let test_dir = Path::new(file!()).parent().unwrap();
        test_dir.join(Path::new("test-data")).join(Path::new(&path))
    }

    #[test]
    fn ten_exemplars() {
        let ap = AffinityPropagation::<f32>::new(0.5, 4, 10, 4000);
        run_test(
            &ap,
            NegEuclidean::default(),
            file(&"near-exemplar-10.test"),
            Value(-1000.),
            0.99,
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
            0.99,
        );
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
            0.97,
        )
    }

    /// This test is very long-running. Run in release mode to reduce clock time.
    #[test]
    fn binsanity2() {
        let ap = AffinityPropagation::<f32>::new(0.95, 4, 400, 4000);
        run_test(
            &ap,
            NegEuclidean::default(),
            file(&"binsanity.2.test"),
            Value(-10.),
            0.98,
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
            0.98,
            0.98,
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
            0.98,
            0.98,
        )
    }

    /// Very low ARI
    #[test]
    #[should_panic]
    fn breast_cancer() {
        let ap = AffinityPropagation::<f32>::new(0.95, 4, 400, 4000);
        run_test(
            &ap,
            LogEuclidean::default(),
            file(&"breast_cancer.test"),
            Value(-10000.),
            0.98,
            0.98,
        )
    }
}
