use crate::affinity_propagation::{AffinityPropagation, Config, Euclidean};
use crate::ops::from_file;
use std::path::Path;

mod affinity_propagation;
mod ops;

#[macro_use]
extern crate clap;

fn main() {
    let matches = clap_app!(myapp =>
        (version: "1.0")
        (author: "Chris N. <christopher.neely1200@gmail.com>")
        (about: "Vectorized and Parallelized Affinity Propagation")
        (@arg INPUT: -i --input +takes_value +required "Set path to input file")
        (@arg PREF: -p --preference +takes_value "Set preference")
        (@arg MAX_ITER: -m --max-iter +takes_value "Maximum iterations")
        (@arg CONV_ITER: -c --convergence-iter +takes_value "Convergence iterations")
        (@arg DAMPING: -d --dampig +takes_value "Set damping value")
        (@arg THREADS: -t --threads +takes_value "Set number of worker threads")
    )
    .get_matches();

    let input_file = matches.value_of("INPUT").unwrap().to_string();
    let preference = matches
        .value_of("PREF")
        .unwrap_or("-10.")
        .parse::<f32>()
        .unwrap();
    let max_iter = matches
        .value_of("MAX_ITER")
        .unwrap_or("100")
        .parse::<usize>()
        .unwrap();
    let conv_iter = matches
        .value_of("CONV_ITER")
        .unwrap_or("10")
        .parse::<usize>()
        .unwrap();
    let damping = matches
        .value_of("DAMPING")
        .unwrap_or("0.9")
        .parse::<f32>()
        .unwrap();
    let threads = matches
        .value_of("THREADS")
        .unwrap_or("4")
        .parse::<usize>()
        .unwrap();
    let (x, y) = from_file(Path::new(&input_file).to_path_buf());
    AffinityPropagation::predict(
        x,
        y,
        Config {
            preference,
            max_iterations: max_iter,
            convergence_iter: conv_iter,
            damping,
            workers: threads,
        },
        Euclidean {},
    );
}
