#[macro_use]
extern crate clap;

use std::path::Path;
use std::process::exit;

use affinityprop::{AffinityPropagation, NegEuclidean};

use crate::ops::{display_results, from_file};

mod ops;

// TODO: Error output formatting
fn main() {
    let matches = clap_app!(affinityprop =>
        (version: "0.1.0")
        (author: "Chris N. <christopher.neely1200@gmail.com>")
        (about: "Vectorized and Parallelized Affinity Propagation")
        (@arg INPUT: -i --input +takes_value +required "Path to input file")
        (@arg PREF: -p --preference +takes_value +allow_hyphen_values "Non-positive preference, default=median pairwise similarity")
        (@arg MAX_ITER: -m --max_iter +takes_value "Maximum iterations, default=100")
        (@arg CONV_ITER: -c --convergence_iter +takes_value "Convergence iterations, default=10")
        (@arg DAMPING: -d --damping +takes_value "Damping value in range (0, 1), default=0.9")
        (@arg THREADS: -t --threads +takes_value "Number of worker threads, default=4")
        (@arg PRECISION: -r --precision +takes_value "Set f32 or f64 precision, default=f32")
    )
    .get_matches();

    let input_file = matches.value_of("INPUT").unwrap().to_string();
    if !Path::new(&input_file).exists() {
        eprintln!("Unable to locate input file {}", input_file);
        exit(1);
    }
    let max_iterations = matches
        .value_of("MAX_ITER")
        .unwrap_or("100")
        .parse::<usize>()
        .unwrap_or_else(|_| {
            eprintln!("Unable to parse max_iterations");
            exit(1);
        });
    let convergence_iter = matches
        .value_of("CONV_ITER")
        .unwrap_or("10")
        .parse::<usize>()
        .unwrap_or_else(|_| {
            eprintln!("Unable to parse convergence_iter");
            exit(1);
        });
    let threads = matches
        .value_of("THREADS")
        .unwrap_or("4")
        .parse::<usize>()
        .unwrap_or_else(|_| {
            eprintln!("Unable to parse threads");
            exit(1);
        });
    let precision = matches.value_of("PRECISION").unwrap_or("f32");
    let preference = matches.value_of("PREF");
    let preference = match preference {
        Some(p) => {
            let p = p.parse::<f64>().unwrap_or_else(|_| {
                eprintln!("Unable to parse preference");
                exit(1);
            });
            if p > 0. {
                eprintln!("Preference must be non-positive");
                exit(2);
            }
            Some(p)
        }
        None => None,
    };

    let damping = matches
        .value_of("DAMPING")
        .unwrap_or("0.9")
        .parse::<f32>()
        .unwrap_or_else(|_| {
            eprintln!("Unable to parse damping");
            exit(1);
        });
    if damping <= 0. || damping >= 1. {
        eprintln!("Improper parameter set!");
        exit(2);
    }
    // Validate values
    if threads < 1 || convergence_iter < 1 || max_iterations < 1 {
        eprintln!("Improper parameter set!");
        exit(2);
    }
    // Run AP
    match precision {
        "f64" => {
            let (x, y) = from_file::<f64>(Path::new(&input_file).to_path_buf());
            let ap = AffinityPropagation::new(
                preference,
                damping as f64,
                threads,
                convergence_iter,
                max_iterations,
            );
            let (converged, results) = ap.predict(&x, NegEuclidean::default());
            display_results(converged, &results, &y);
        }
        _ => {
            let (x, y) = from_file::<f32>(Path::new(&input_file).to_path_buf());
            let preference = match preference {
                Some(p) => Some(p as f32),
                None => None,
            };
            let ap = AffinityPropagation::new(
                preference,
                damping,
                threads,
                convergence_iter,
                max_iterations,
            );
            let (converged, results) = ap.predict(&x, NegEuclidean::default());
            display_results(converged, &results, &y);
        }
    };
}
