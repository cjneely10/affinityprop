use crate::affinity_propagation::{AffinityPropagation, Config, Euclidean};
use ndarray::arr2;

mod affinity_propagation;

fn main() {
    println!("Hello, world!");
    let x = arr2(&[[1., 2., 3.], [4., 5., 6.]]);
    let y = vec!['a', 'b'];
    AffinityPropagation::predict(x, y, Config::default(), Euclidean);
}
