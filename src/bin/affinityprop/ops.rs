use ndarray::{Array2, Axis};
use num_traits::Float;
use std::fmt::Debug;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::str::FromStr;

/// Reads in a file formatted as (tab separated):
///     id1 val1 val2 val3
///     id2 val1 val2 val3
///
/// Provide as many ids and values as desired
/// All rows should be same length
/// Values should be floating-point decimal values
pub(crate) fn from_file<F>(p: PathBuf) -> (Array2<F>, Vec<String>)
where
    F: Float + Default + FromStr,
    <F as FromStr>::Err: Debug,
{
    let reader = BufReader::new(File::open(p).unwrap());
    let mut labels = Vec::new();
    let mut data = Vec::new();
    // Read tab-delimited file
    reader.lines().map(|l| l.unwrap()).for_each(|line| {
        let mut line = line.split('\t');
        // ID as first col
        labels.push(line.next().expect("Error loading line label").to_string());
        // Rest are data
        data.push(
            line.map(|s| s.parse::<F>().expect("Error parsing data in file"))
                .collect::<Vec<F>>(),
        );
    });
    // Validate data was loaded
    assert!(
        data.len() > 1,
        "Data file is empty or only contains a single entry"
    );
    // Validate data all has same length
    let length = data[0].len();
    data.iter().skip(1).for_each(|v| {
        assert_eq!(v.len(), length, "Input data rows must all be same length!");
    });
    // Convert data to Array2
    let mut out = Array2::<F>::default((data.len(), data[0].len()));
    out.axis_iter_mut(Axis(0))
        .enumerate()
        .for_each(|(idx1, mut row)| {
            row.iter_mut().enumerate().for_each(|(idx2, col)| {
                *col = data[idx1][idx2];
            });
        });
    (out, labels)
}
