use crate::affinity_propagation::{AffinityPropagation, Config};

mod affinity_propagation;

fn main() {
    println!("Hello, world!");
    AffinityPropagation::begin(
        60000,
        Config {
            workers: 16,
            damping: 0.1,
        },
        3,
    );
}
