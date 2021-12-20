use crate::affinity_propagation::{AffinityPropagation, Config, Euclidean};
use ndarray::{arr2, Array2};

mod affinity_propagation;

fn main() {
    println!("Hello, world!");
    let x = arr2(&[[1., 2., 3.], [4., 5., 6.]]);
    let y = vec![0, 1];
    AffinityPropagation::predict(x, y, Config::default(), Euclidean);
}
