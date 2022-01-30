#[macro_use]
extern crate clap;

use std::collections::HashMap;
use std::path::Path;
use std::process::exit;

use ndarray::Array2;
use num_traits::Float;

use affinityprop::{AffinityPropagation, LogEuclidean, NegCosine, NegEuclidean, Preference};

use crate::ops::{display_results, from_file};

mod ops;

#[cfg(not(tarpaulin_include))]
fn main() {
    let matches = clap_app!(affinityprop =>
        (version: "0.1.0")
        (author: "Chris N. <christopher.neely1200@gmail.com>")
        (about: "Vectorized and Parallelized Affinity Propagation")
        (@arg INPUT: -i --input +takes_value +required "Path to input file")
        (@arg DELIMITER: -l --delimiter +takes_value "File delimiter, default '\\t'")
        (@arg PREF: -p --preference +takes_value +allow_hyphen_values "Preference to be own exemplar, default=median pairwise similarity")
        (@arg MAX_ITER: -m --max_iter +takes_value "Maximum iterations, default=100")
        (@arg CONV_ITER: -c --convergence_iter +takes_value "Convergence iterations, default=10")
        (@arg DAMPING: -d --damping +takes_value "Damping value in range (0, 1), default=0.9")
        (@arg THREADS: -t --threads +takes_value "Number of worker threads, default=4")
        (@arg PRECISION: -r --precision +takes_value "Set f32 or f64 precision, default=f32")
        (@arg SIMILARITY: -s --similarity +takes_value "Set similarity calculation method (0=NegEuclidean,1=NegCosine,2=LogEuclidean,3=precalculated), default=0")
    )
    .get_matches();

    // Parse
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
    let preference = match matches.value_of("PREF") {
        Some(p) => {
            let p = p.parse::<f32>().unwrap_or_else(|_| {
                eprintln!("Unable to parse preference");
                exit(1);
            });
            Some(p)
        }
        None => None,
    };
    let similarity: usize = match matches.value_of("SIMILARITY") {
        Some(s) => {
            let s = s.parse::<usize>().unwrap_or_else(|_| {
                eprintln!("Unable to parse similarity");
                exit(1);
            });
            if s > 3 {
                eprintln!("Invalid similarity selected");
                exit(2);
            };
            s
        }
        None => 0,
    };
    let damping = matches
        .value_of("DAMPING")
        .unwrap_or("0.9")
        .parse::<f32>()
        .unwrap_or_else(|_| {
            eprintln!("Unable to parse damping");
            exit(1);
        });
    let delimiter = matches.value_of("DELIMITER").unwrap_or("\t");

    // Validate values
    if damping <= 0. || damping >= 1. {
        eprintln!("Improper parameter set!");
        exit(2);
    }
    if threads < 1 || convergence_iter < 1 || max_iterations < 1 {
        eprintln!("Improper parameter set!");
        exit(2);
    }
    // Run AP
    let is_precalculated = similarity == 3;
    match precision {
        "f64" => match from_file::<f64>(
            Path::new(&input_file).to_path_buf(),
            delimiter,
            is_precalculated,
        ) {
            Ok((x, y)) => {
                let preference = preference.map(|p| p as f64);
                let ap = AffinityPropagation::new(
                    damping as f64,
                    threads,
                    convergence_iter,
                    max_iterations,
                );
                predict(&ap, &similarity, x, y, preference);
            }
            Err(e) => {
                eprintln!("{}", e.message);
                exit(3);
            }
        },
        _ => match from_file::<f32>(
            Path::new(&input_file).to_path_buf(),
            delimiter,
            is_precalculated,
        ) {
            Ok((x, y)) => {
                let ap =
                    AffinityPropagation::new(damping, threads, convergence_iter, max_iterations);
                predict(&ap, &similarity, x, y, preference);
            }
            Err(e) => {
                eprintln!("{}", e.message);
                exit(3);
            }
        },
    };
}

/// Run predictor with specified similarity metric
#[cfg(not(tarpaulin_include))]
fn predict<F>(
    ap: &AffinityPropagation<F>,
    similarity: &usize,
    x: Array2<F>,
    y: Vec<String>,
    preference: Option<F>,
) where
    F: Float + Send + Sync,
{
    let preference = match preference {
        Some(pref) => Preference::Value(pref),
        None => Preference::Median,
    };
    let a: (bool, HashMap<usize, Vec<usize>>);
    match similarity {
        1 => {
            a = ap.predict(&x, NegCosine::default(), preference);
        }
        2 => {
            a = ap.predict(&x, LogEuclidean::default(), preference);
        }
        3 => {
            a = ap.predict_precalculated(x, preference);
        }
        _ => {
            a = ap.predict(&x, NegEuclidean::default(), preference);
        }
    }
    display_results(a.0, &a.1, y);
}
