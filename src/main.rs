use crate::affinity_propagation::{AffinityPropagation, Config, Euclidean};
use crate::ops::from_file;
use std::path::Path;

mod affinity_propagation;
mod ops;

#[macro_use]
extern crate clap;

fn main() {
    let matches = clap_app!(myapp =>
        (version: "0.1.0")
        (author: "Chris N. <christopher.neely1200@gmail.com>")
        (about: "Vectorized and Parallelized Affinity Propagation")
        (@arg INPUT: -i --input +takes_value +required "Set path to input file")
        (@arg PREF: -p --preference +takes_value +allow_hyphen_values "Set preference")
        (@arg MAX_ITER: -m --max_iter +takes_value "Maximum iterations")
        (@arg CONV_ITER: -c --convergence_iter +takes_value "Convergence iterations")
        (@arg DAMPING: -d --damping +takes_value "Set damping value")
        (@arg THREADS: -t --threads +takes_value "Set number of worker threads")
    )
    .get_matches();

    let input_file = matches.value_of("INPUT").unwrap().to_string();
    let preference = matches
        .value_of("PREF")
        .unwrap_or("-10.")
        .parse::<f32>()
        .unwrap();
    let mut max_iterations = matches
        .value_of("MAX_ITER")
        .unwrap_or("100")
        .parse::<usize>()
        .unwrap();
    let convergence_iter = matches
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
    if convergence_iter > max_iterations {
        max_iterations = convergence_iter;
    }
    let (x, y) = from_file(Path::new(&input_file).to_path_buf());
    let cfg = Config {
        preference,
        max_iterations,
        convergence_iter,
        damping,
        threads,
    };
    println!("{:?}", cfg);
    AffinityPropagation::predict(x, y, cfg, Euclidean::default());
}
