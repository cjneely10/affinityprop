#[cfg(test)]
mod test {
    use affinityprop::{AffinityPropagation, Euclidean};
    use ndarray::{arr2, Array2};

    #[test]
    fn simple() {
        let x: Array2<f32> = arr2(&[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]);
        let y = vec!["1".to_string(), "2".to_string(), "3".to_string()];
        let mut ap =
            AffinityPropagation::with_defaults(x, &y, Euclidean::default(), false);
        ap.predict();
        assert!(ap.results().len() == 1 && ap.results().contains_key(&1));
    }
}