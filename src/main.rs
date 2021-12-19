use crate::affinity_propagation::AffinityPropagation;

mod affinity_propagation;

fn main() {
    println!("Hello, world!");
    AffinityPropagation::begin(50000, 50000, 16, 3);
}
