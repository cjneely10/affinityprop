//! The `affinityprop` crate provides an optimized implementation of the Affinity Propagation
//! clustering algorithm, which identifies cluster of data without *a priori* knowledge about the
//! number of clusters in the data. The original algorithm was developed by
//! <a href="http://utstat.toronto.edu/reid/sta414/frey-affinity.pdf" target="_blank">Brendan Frey and Delbery Dueck</a>
//!
//! # About
//! Affinity Propagation identifies a subset of representative examples from a dataset, known as
//! **exemplars**.
//!
//! Briefly, the algorithm accepts as input a matrix describing **pairwise similarity** for all data
//! values. This information is used to calculate pairwise **responsibility** and **availability**.
//! Responsibility *`r(i,j)`* describes how well-suited point *`j`* is to act as an exemplar for
//! point *`i`* when compared to other potential exemplars. Availability *`a(i,j)`* describes how
//! appropriate it is for point *i* to accept point *j* as its exemplar when compared to
//! other exemplars.
//!
//! Users provide a number of **convergence iterations** to repeat the calculations, after which the
//! potential exemplars are extracted from the dataset. Then, the algorithm continues to repeat
//! until the exemplar values stop changing, or until the **maximum iterations** are met.
//!
//! # Why this crate?
//! The nature of Affinity Propagation demands an *O(n<sup>2</sup>)* runtime. An existing
//! <a href="https://github.com/scikit-learn/scikit-learn/blob/0d378913b/sklearn/cluster/_affinity_propagation.py#L38" target="_blank">sklearn</a>
//! version is implemented using the Python library
//! <a href="https://numpy.org/doc/stable/index.html" target="_blank">numpy</a>
//! which incorporates vectorized row operations. Coupled with **SIMD** instructions, this results
//! in decreased time to finish.
//!
//! However, in applications with large input values, the *O(n<sup>2</sup>)* runtime is still
//! prohibitive. This crate implements Affinity Propagation using the
//! <a href="https://crates.io/crates/rayon" target="_blank">rayon</a>
//! crate, which allows for a drastic decrease in overall runtime - as much as 30-60% when compiled
//! in release mode!
//!
//! # Dependencies
//! <a href="https://doc.rust-lang.org/cargo/getting-started/installation.html">cargo</a>
//! with `rustc >=1.58`
//!
//! # Installation
//! ## In Rust code
//! ```toml
//! [dependencies]
//! affinityprop = { git = "https://github.com/cjneely10/affinityprop", version = "0.1.1" }
//! ndarray = "0.15.4"
//! ```
//!
//! ## As a command-line tool
//! ```shell
//! cargo install affinityprop --git https://github.com/cjneely10/affinityprop --version 0.1.1
//! ```
//!
//! # Usage
//!
//! ## From Rust code
//!
//! The `affinityprop` crate expects a type that defines how to calculate pairwise [`Similarity`]
//! for all data points. This crate provides the [`NegEuclidean`], [`NegCosine`], and
//! [`LogEuclidean`] structs, which are defined as `-1 * sum((a - b)**2)`, `-1 * (a . b)/(|a|*|b|)`,
//! and `sum(log((a - b)**2))`, respectively.
//!
//! Users who wish to calculate similarity differently are advised that **Affinity Propagation
//! expects *s(i,j)* > *s(i, k)* iff *i* is more similar to *j* than it is to *k***.
//!
//!     use ndarray::{arr1, arr2, Array2};
//!     use affinityprop::{AffinityPropagation, NegCosine, Preference};
//!     
//!     let x: Array2<f32> = arr2(&[[0., 1., 0.], [2., 3., 2.], [3., 2., 3.]]);
//!
//!     // Cluster using negative cosine similarity with a pre-defined preference
//!     let ap = AffinityPropagation::default();
//!     let (converged, results) = ap.predict(&x, NegCosine::default(), Preference::Value(-10.));
//!     assert!(converged && results.len() == 1 && results.contains_key(&0));
//!
//!     // Cluster with list of preference values
//!     let pref = arr1(&[0., -1., 0.]);
//!     let (converged, results) = ap.predict(&x, NegCosine::default(), Preference::List(&pref));
//!     assert!(converged);
//!     assert!(results.len() == 2 && results.contains_key(&0) && results.contains_key(&2));
//!
//!     // Use damping=0.5, threads=2, convergence_iter=10, max_iterations=100,
//!     // median similarity as preference
//!     let ap = AffinityPropagation::new(0.5, 2, 10, 100);
//!     let (converged, results) = ap.predict(&x, NegCosine::default(), Preference::Median);
//!     assert!(converged);
//!     assert!(results.len() == 2 && results.contains_key(&0) && results.contains_key(&2));
//!
//!     // Predict with pre-calculated similarity
//!     let s: Array2<f32> = arr2(&[[0., -3., -12.], [-3., 0., -3.], [-12., -3., 0.]]);
//!     let ap = AffinityPropagation::default();
//!     let (converged, results) = ap.predict_precalculated(s, Preference::Value(-10.));
//!     assert!(converged && results.len() == 1 && results.contains_key(&1));
//!
//! ## From the Command Line
//!
//! `affinityprop` can be run from the command-line and used to analyze a file of data:
//!
//! ```text
//! ID1  val1  val2
//! ID2  val3  val4
//! ...
//! ```
//!
//! where ID*n* is any string identifier and val*n* are floating-point (decimal) values. The file
//! delimiter is provided from the command line.
//!
//! Users may provide a pre-calculated similarity matrix in the same manner:
//!
//! ```text
//! val1  val2
//! val3  val4
//! ...
//! ```
//!
//! IDs will automatically be assigned by zero-based index.
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
//!     -d, --damping <DAMPING>               Damping value in range (0, 1), default=0.9
//!     -l, --delimiter <DELIMIER>            File delimiter, default '\t'
//!     -i, --input <INPUT>                   Path to input file
//!     -m, --max_iter <MAX_ITER>             Maximum iterations, default=100
//!     -r, --precision <PRECISION>           Set f32 or f64 precision, default=f32
//!     -p, --preference <PREF>               Preference to be own exemplar, default=median pairwise similarity
//!     -s, --similarity <SIMILARITY>         Set similarity metric (0=NegEuclidean,1=NegCosine,2=LogEuclidean,3=precalculated), default=0
//!     -t, --threads <THREADS>               Number of worker threads, default=4
//! ```
//!
//! ### Results
//! Results are printed to stdout in the format:
//!
//! ```text
//! Converged=true/false nClusters=NC nSamples=NS
//! >Cluster=n size=N exemplar=i
//! [comma-separated cluster member indices]
//! >Cluster=n size=N exemplar=i
//! [comma-separated cluster member indices]
//! ...
//! ```
//!
//! # Runtime and Resource Notes
//!
//! Affinity Propagation is *O(n<sup>2</sup>)* in both runtime and memory.
//! This crate seeks to address the former, not the latter.
//!
pub use affinity_propagation::{AffinityPropagation, Cluster, ClusterMap, ClusterResults, Idx};
pub use preference::Preference;
pub use similarity::{LogEuclidean, NegCosine, NegEuclidean, Similarity};

mod affinity_propagation;
mod algorithm;
mod preference;
mod similarity;
