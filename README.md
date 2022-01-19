[![Rust](https://github.com/cjneely10/affinityprop/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/cjneely10/affinityprop/actions/workflows/rust.yml)
[![GitHub](https://img.shields.io/github/license/cjneely10/affinityprop)](https://www.gnu.org/licenses/gpl-3.0.html)
[![affinityprop: rustc 1.58](https://img.shields.io/badge/affinityprop-rustc__1.58-blue)](https://doc.rust-lang.org/rustc/what-is-rustc.html)
![coverage](https://img.shields.io/badge/coverage-93%25-success)

# AffinityProp
The `affinityprop` crate provides an optimized implementation of the Affinity Propagation
clustering algorithm, which identifies cluster of data without *a priori* knowledge about the
number of clusters in the data. The original algorithm was developed by
<a href="http://utstat.toronto.edu/reid/sta414/frey-affinity.pdf" target="_blank">Brendan Frey and Delbery Dueck</a>

# About
Affinity Propagation identifies a subset of representative examples from a dataset, known as
**exemplars**.

Briefly, the algorithm accepts as input a matrix describing **pairwise similarity** for all data
values. This information is used to calculate pairwise **responsibility** and **availability**.
Responsibility *`r(i,j)`* describes how well-suited point *`j`* is to act as an exemplar for
point *`i`* when compared to other potential exemplars. Availability *`a(i,j)`* describes how
appropriate it is for point *i* to accept point *j* as its exemplar when compared to
other exemplars.

Users provide a number of **convergence iterations** to repeat the calculations, after which the
potential exemplars are extracted from the dataset. Then, the algorithm continues to repeat
until the exemplar values stop changing, or until the **maximum iterations** are met.

# Why this crate?
The nature of Affinity Propagation demands an *O(n<sup>2</sup>)* runtime. An existing
<a href="https://github.com/scikit-learn/scikit-learn/blob/0d378913b/sklearn/cluster/_affinity_propagation.py#L38" target="_blank">sklearn</a>
version is implemented using the Python library
<a href="https://numpy.org/doc/stable/index.html" target="_blank">numpy</a>
which incorporates vectorized row operations. Coupled with **SIMD** instructions, this results
in decreased time to finish.

However, in applications with large input values, the *O(n<sup>2</sup>)* runtime is still
prohibitive. This crate implements Affinity Propagation using the
<a href="https://crates.io/crates/rayon" target="_blank">rayon</a>
crate, which allows for a drastic decrease in overall runtime - as much as 30-60% when compiled
in release mode!

# Installation
## In Rust code
```toml
[dependencies]
affinityprop = { git = "https://github.com/cjneely10/affinityprop", version = "0.1.0" }
```

## As a command-line tool
```shell
cargo install --git https://github.com/cjneely10/affinityprop
```

# Usage

## From Rust code

The `affinityprop` crate expects a type that defines how to calculate pairwise `Similarity`
for all data points. This crate provides the `NegEuclidean`, `NegCosine`, and
`LogEuclidean` structs, which are defined as `-1 * sum((a - b)**2)`, `-1 * (a . b)/(|a|*|b|)`,
and `sum(log((a - b)**2))`, respectively.

Users who wish to calculate similarity differently are advised that **Affinity Propagation
expects *s(i,j)* > *s(i, k)* iff *i* is more similar to *j* than it is to *k***.

```rust
use ndarray::{arr2, Array2};
use affinityprop::{AffinityPropagation, NegCosine};
let x: Array2<f32> = arr2(&[[0., 1., 0.], [2., 3., 2.], [3., 2., 3.]]);

// Cluster using negative cosine similarity
let ap = AffinityPropagation::default();
let (converged, results) = ap.predict(&x, NegCosine::default());
assert!(converged && results.len() == 1 && results.contains_key(&0));

// Use median similarity as preference, damping=0.5, threads=2,
// convergence_iter=10, max_iterations=100
let ap = AffinityPropagation::new(None, 0.5, 2, 10, 100);
let (converged, results) = ap.predict(&x, NegCosine::default());
assert!(converged && results.len() == 2);
assert!(results.contains_key(&0) && results.contains_key(&2));
```

## From the Command Line

`affinityprop` can be run from the command-line and used to analyze a tab-delimited
file of data:

```text
ID1  val1  val2
ID2  val3  val4
...
```

where ID*n* is any string identifier and val*n* are floating-point (decimal) values.

### Help Menu
```text
affinityprop 0.1.0
Chris N. <christopher.neely1200@gmail.com>
Vectorized and Parallelized Affinity Propagation

USAGE:
    affinityprop [OPTIONS] --input <INPUT>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -c, --convergence_iter <CONV_ITER>    Convergence iterations, default=10
    -d, --damping <DAMPING>               Damping value in range (0, 1), default=0.9
    -l, --delimiter <DELIMIER>            File delimiter, default '\t'
    -i, --input <INPUT>                   Path to input file
    -m, --max_iter <MAX_ITER>             Maximum iterations, default=100
    -r, --precision <PRECISION>           Set f32 or f64 precision, default=f32
    -p, --preference <PREF>               Preference to be own exemplar, default=median pairwise similarity
    -s, --similarity <SIMILARITY>         Set similarity metric (0=NegEuclidean,1=NegCosine,2=LogEuclidean), default=0
    -t, --threads <THREADS>               Number of worker threads, default=4
```

### Results
Results are printed to stdout in the format:

```text
Converged=true/false nClusters=NC nSamples=NS
>Cluster=n size=N exemplar=i
[comma-separated cluster members]
>Cluster=n size=N exemplar=i
[comma-separated cluster members]
...
```

# Runtime and Resource Notes

Affinity Propagation is *O(n<sup>2</sup>)* in both runtime and memory.
This crate seeks to address the former, not the latter.

