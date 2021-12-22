# AffinityProp
Vectorized and Parallelized Affinity Propagation

## About

`affinityprop` provides a command-line accessible interface for running the Affinity Propagation
clustering algorithm.

The implementation largely mimics the [sklearn version](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html),
but has been implemented in Rust using the `rayon` and `ndarray` packages to allow for parallel iteration.

## Installation

Download this repository

```shell
git clone git@github.com:cjneely10/affinityprop.git
cd affinityprop
```

If desired, increase precision by changing line 9 of the `Cargo.toml` file from `f32` to `f64`. 

Compile using `cargo`:

```shell
cargo build --release
```

The program binary will be present in the `target/release` directory.

## Usage

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
    -p, --preference <PREF>               Preference, default=-10.0
    -t, --threads <THREADS>               Number of worker threads, default=4
```

## Example

Provide a **tab-separated** file in the format:

```text
ID1  val1  val2
ID2  val3  val4
...
```

where ID*n* is any string identifier and val*n* are floating-point (decimal) values.

We have provided an example file in the `test` directory:

```shell
./target/release/affinityprop -i ./test/Infant_gut_assembly.cov.x100.lognorm -c 400 -m 4000 -p -10.0 -d 0.95 -t 16 > ./test/output-file.txt
```

### Results

Increasing thread count can see up to a 60% runtime reduction for 32-bit floating point precision,
and around a 20-30% reduction in 64-bit mode.

Results are printed to stdout in the format:

```text
Config { ... }
Converged=true/false nClusters=NC nSamples=NS
>Cluster=n size=N exemplar=i
[comma-separated cluster members]
>Cluster=n size=N exemplar=i
[comma-separated cluster members]
...
```

### Notes

Affinity Propagation is *O(n<sup>2</sup>)* in both runtime and memory. 
This package seeks to address the former, not the latter. 
