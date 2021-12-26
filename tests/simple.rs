#[cfg(test)]
mod test {
    use affinityprop::{AffinityPropagation, Euclidean};
    use ndarray::{arr2, Array2};

    #[test]
    fn simple() {
        let x: Array2<f32> = arr2(&[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]);
        let y = vec!["1", "2", "3"];
        let mut ap = AffinityPropagation::default();
        let results = ap.predict(x, &y, Euclidean::default());
        assert!(results.len() == 1 && results.contains_key(&"2"));
        ap.display_results(&results);
    }
}
