#[cfg(test)]
mod test {
    use affinityprop::{AffinityPropagation, Config, Euclidean, Value};
    use ndarray::{arr2, Array2};

    #[test]
    fn init() {
        let x: Array2<Value> = arr2(&[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]);
        let y = vec!["1".to_string(), "2".to_string(), "3".to_string()];
        AffinityPropagation::predict(x, y, Config::default(), Euclidean::default());
    }
}
