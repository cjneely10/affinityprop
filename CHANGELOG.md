# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

### Python bindings
- [PyO3](https://github.com/PyO3/pyo3)-derived bindings for use within Python

## [0.3.0-alpha] - 2022-07-01
### Changed
- NaN values are supported in input matrices
  - Previously, cli would exit on nan detection
  - Now, notify user via stderr
    - Only exit if full row of NaN found
  - Provided similarity calculations now handle NaN values by replacing them with `0` prior to final sum operations
    - See `NegEuclidean`'s `Similarity` implementation
- NaN check in `predict_precalculated`
  - Panic on detection

## [0.2.0] - 2022-06-24
### Added
- Better stderr messages on file-parsing errors
  - Line/col numbers and kind of error

### Changed
- Small edits to pass-by-value instead of by-reference

### Fixed
- `nan` values were accepted as valid input.
  - No valid similarity calculation is present for `nan` values

## [0.1.1] - 2022-02-15
First release of `affinityprop`
