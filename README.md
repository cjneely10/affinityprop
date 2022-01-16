[![Rust](https://github.com/cjneely10/affinityprop/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/cjneely10/affinityprop/actions/workflows/rust.yml)
[![GitHub](https://img.shields.io/github/license/cjneely10/affinityprop)](https://www.gnu.org/licenses/gpl-3.0.html)
![affinityprop: rustc 1.53+](https://img.shields.io/badge/affinityprop-rustc__1.53+-blue)

# AffinityProp
Vectorized and Parallelized Affinity Propagation

The `affinityprop` crate provides an optimized implementation of the Affinity Propagation
clustering algorithm, which identifies cluster of data without *a priori* knowledge about the
number of clusters in the data.

# About
Affinity Propagation identifies a subset of representative examples from a dataset, known as
**exemplars**. The original algorithm was developed by [Brendan Frey and Delbery Dueck](http://utstat.toronto.edu/reid/sta414/frey-affinity.pdf).

Briefly, the algorithm accepts as input a matrix describing **pairwise similarity** for all data
values. This information is used to calculate pairwise. **responsibility** and **availability**.
Responsibility *r(i,j)* describes how well-suited point *j* is to act as an exemplar for
point *i* when compared to other potential exemplars. Availability *a(i,j)* describes how
appropriate is the selection of point *j* to be the exemplar for point *i* when compared to
other exemplars.

Users provide a number of **convergence iterations** to repeat the calculations, after which the
potential exemplars are extracted from the dataset. Then, the algorithm continues to repeat
until the exemplar values stop changing, or the **maximum iterations** are met.

# Why this crate?
The nature of Affinity Propagation demands an *O(n<sup>2</sup>)* runtime. An existing [sklearn](https://github.com/scikit-learn/scikit-learn/blob/0d378913b/sklearn/cluster/_affinity_propagation.py#L38)
implementation is implemented using the Python library [numpy](https://numpy.org/doc/stable/index.html),
which implements vectorized calculations. Coupled with **SIMD** instructions, this results in
decreased user runtime.

However, in applications with large input values, the *O(n<sup>2</sup>)* runtime is still
prohibitive. This crate implements Affinity Propagation using the Rust [rayon](https://crates.io/crates/rayon)
library, which allows for a drastic decrease in overall runtime - as much as 30-60% depending
on floating point precision!

## Installation

Download this repository

```shell
git clone git@github.com:cjneely10/affinityprop.git
cd affinityprop
```

## Usage

Run the binary using either Rust's package manager:

```shell
cargo run --release --bin affinityprop -- -h
```

Or, compile using `cargo`:

```shell
cargo build --release
```

And then run directly:

```shell
./target/release/affinityprop -h
```

### Help menu

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
    -d, --damping <DAMPING>               Damping value, default=0.9
    -i, --input <INPUT>                   Path to input file
    -m, --max_iter <MAX_ITER>             Maximum iterations, default=100
    -r, --precision <PRECISION>           Set f32 or f64 precision, default=f32
    -p, --preference <PREF>               Preference, default=-10.0
    -t, --threads <THREADS>               Number of worker threads, default=4
```

### Command-line input data

Provide a **tab-separated** file in the format:

```text
ID1  val1  val2
ID2  val3  val4
...
```

where ID*n* is any string identifier and val*n* are floating-point (decimal) values.

## Example

We have provided an example file in the `data` directory:

```shell
./target/release/affinityprop -i ./data/Infant_gut_assembly.cov.x100.lognorm -c 400 -m 4000 -p -10.0 -d 0.95 -t 16 > output-file.txt
```

-or-

```shell
cargo run --release --bin affinityprop -- -i ./data/Infant_gut_assembly.cov.x100.lognorm -c 400 -m 4000 -p -10.0 -d 0.95 -t 16 > output-file.txt
```

### Results

Increasing thread count can see up to a 60% runtime reduction for 32-bit floating point precision,
and around a 20-30% reduction in 64-bit mode (i.e., when run using the `-r f64` flag).

Results are printed to stdout in the format:

```text
Converged=true/false nClusters=NC nSamples=NS
>Cluster=n size=N exemplar=i
[comma-separated cluster members]
>Cluster=n size=N exemplar=i
[comma-separated cluster members]
...
```

### Runtime and Resource Notes

Affinity Propagation is *O(n<sup>2</sup>)* in both runtime and memory. 
This crate seeks to address the former, not the latter.
