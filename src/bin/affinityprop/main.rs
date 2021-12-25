use crate::ops::from_file;
use affinityprop::{AffinityPropagation, Config, Euclidean, Value};
use std::path::Path;
use std::process::exit;

mod ops;

#[macro_use]
extern crate clap;

fn main() {
    let matches = clap_app!(affinityprop =>
        (version: "0.1.0")
        (author: "Chris N. <christopher.neely1200@gmail.com>")
        (about: "Vectorized and Parallelized Affinity Propagation")
        (@arg INPUT: -i --input +takes_value +required "Path to input file")
        (@arg PREF: -p --preference +takes_value +allow_hyphen_values "Preference, default=-10.0")
        (@arg MAX_ITER: -m --max_iter +takes_value "Maximum iterations, default=100")
        (@arg CONV_ITER: -c --convergence_iter +takes_value "Convergence iterations, default=10")
        (@arg DAMPING: -d --damping +takes_value "Damping value, default=0.9")
        (@arg THREADS: -t --threads +takes_value "Number of worker threads, default=4")
    )
    .get_matches();

    let input_file = matches.value_of("INPUT").unwrap().to_string();
    if !Path::new(&input_file).exists() {
        eprintln!("Unable to locate input file {}", input_file);
        exit(1);
    }
    let preference = matches
        .value_of("PREF")
        .unwrap_or("-10.0")
        .parse::<Value>()
        .expect("Unable to parse preference");
    let mut max_iterations = matches
        .value_of("MAX_ITER")
        .unwrap_or("100")
        .parse::<usize>()
        .expect("Unable to parse max_iterations");
    let convergence_iter = matches
        .value_of("CONV_ITER")
        .unwrap_or("10")
        .parse::<usize>()
        .expect("Unable to parse convergence_iter");
    let damping = matches
        .value_of("DAMPING")
        .unwrap_or("0.9")
        .parse::<Value>()
        .expect("Unable to parse damping");
    let threads = matches
        .value_of("THREADS")
        .unwrap_or("4")
        .parse::<usize>()
        .expect("Unable to parse threads");
    // Validate values
    if threads < 1 || (damping < 0. || damping > 1.) || convergence_iter < 1 || max_iterations < 1 {
        eprintln!("Improper parameter set!");
        exit(2);
    }
    if convergence_iter > max_iterations {
        max_iterations = convergence_iter * 2;
    }
    let (x, y) = from_file(Path::new(&input_file).to_path_buf());

    // Run AP
    let cfg = Config {
        preference,
        max_iterations,
        convergence_iter,
        damping,
        threads,
    };
    println!("{:?}", cfg);
    let mut ap = AffinityPropagation::new(x, &y, cfg, Euclidean::default(), true);
    ap.predict();
}
