//! The `affinityprop` crate provides an optimized implementation of the Affinity Propagation
//! clustering algorithm, which identifies cluster of data without *a priori* knowledge about the
//! number of clusters in the data.
//!
//! # About
//! Affinity Propagation identifies a subset of representative examples from a dataset, known as
//! **exemplars**. The original algorithm was developed by [Brendan Frey and Delbery Dueck](http://utstat.toronto.edu/reid/sta414/frey-affinity.pdf).
//!
//! Briefly, the algorithm accepts as input a matrix describing **pairwise similarity** for all data
//! values. This information is used to calculate pairwise. **responsibility** and **availability**.
//! Responsibility *`r(i,j)`* describes how well-suited point *`j`* is to act as an exemplar for
//! point *`i`* when compared to other potential exemplars. Availability *`a(i,j)`* describes how
//! appropriate is the selection of point *`j`* to be the exemplar for point *`i`* when compared to
//! other exemplars.
//!
//! Users provide a number of **convergence iterations** to repeat the calculations, after which the
//! potential exemplars are extracted from the dataset. Then, the algorithm continues to repeat
//! until the exemplar values stop changing, or the **maximum iterations** are met.
//!
//! # Why this crate?
//! The nature of Affinity Propagation demands an *O(n<sup>2</sup>)* runtime. An existing [sklearn](https://github.com/scikit-learn/scikit-learn/blob/0d378913b/sklearn/cluster/_affinity_propagation.py#L38)
//! implementation is implemented using the Python library [numpy](https://numpy.org/doc/stable/index.html),
//! which implements vectorized calculations. Coupled with **SIMD** instructions, this results in
//! decreased user runtime.
//!
//! However, in applications with large input values, the *O(n<sup>2</sup>)* runtime is still
//! prohibitive. This crate implements Affinity Propagation using the Rust [rayon](https://crates.io/crates/rayon)
//! library, which allows for a drastic decrease in overall runtime - as much as 30-60% depending
//! on floating point precision!
//!
//! # Usage
//!
//! ## From Rust code
//!
//! The `affinityprop` crate expects a type that defines how to calculate pairwise [`Similarity`]
//! for all data points. This crate provides the [`NegEuclidean`] struct that determines the
//! negative Euclidean distance between each point, which is defined as
//! `-1 * sum((point_i - point_j)**2)`.
//!
//!     use ndarray::{arr2, Array2};
//!     use affinityprop::{AffinityPropagation, NegEuclidean};
//!     let x: Array2<f32> = arr2(&[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]);
//!     let ap = AffinityPropagation::default();
//!     let (converged, results) = ap.predict(&x, NegEuclidean::default());
//!     assert!(converged && results.len() == 1 && results.contains_key(&1));
//!
//! Users who wish to calculate similarity differently are advised that **this implementation of
//! Affinity Propagation is only guaranteed to converge for negative similarity values**.
//!
//! ## From the Command Line
//!
//! As a binary, `affinityprop` can be run from the command-line and used to analyze a tab-delimited
//! file of data:
//!
//! ```text
//! ID1  val1  val2
//! ID2  val3  val4
//! ...
//! ```
//!
//! where ID*n* is any string identifier and val*n* are floating-point (decimal) values.
//!
//!
//! ### Help Menu
//! ```text
//! affinityprop 0.1.0
//! Chris N. <christopher.neely1200@gmail.com>
//! Vectorized and Parallelized Affinity Propagation
//!
//! USAGE:
//!     affinityprop [OPTIONS] --input <INPUT>
//!
//! FLAGS:
//!     -h, --help       Prints help information
//!     -V, --version    Prints version information
//!
//! OPTIONS:
//!     -c, --convergence_iter <CONV_ITER>    Convergence iterations, default=10
//!     -d, --damping <DAMPING>               Damping value, default=0.9
//!     -i, --input <INPUT>                   Path to input file
//!     -m, --max_iter <MAX_ITER>             Maximum iterations, default=100
//!     -r, --precision <PRECISION>           Set f32 or f64 precision, default=f32
//!     -p, --preference <PREF>               Preference, default=-10.0
//!     -t, --threads <THREADS>               Number of worker threads, default=4
//! ```
//!

pub use affinity_propagation::AffinityPropagation;
pub use similarity::{NegEuclidean, Similarity};

mod affinity_propagation;
mod algorithm;
mod similarity;
