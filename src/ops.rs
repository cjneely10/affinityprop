use crate::affinity_propagation::Value;
use ndarray::{Array2, Axis};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

/// Reads in a file formatted as (tab separated):
///     id1 val1 val2 val3
///     id2 val1 val2 val3
///
/// Provide as many ids and values as desired
/// All rows should be same length
/// Values should be floating-point decimal values
pub(crate) fn from_file(p: PathBuf) -> (Array2<Value>, Vec<String>) {
    let reader = BufReader::new(File::open(p).unwrap());
    let mut labels = Vec::new();
    let mut data = Vec::new();
    reader.lines().map(|l| l.unwrap()).for_each(|line| {
        let mut line = line.split('\t');
        labels.push(line.next().expect("Error loading line label").to_string());
        data.push(
            line.map(|s| s.parse::<Value>().expect("Error parsing data in file"))
                .collect::<Vec<Value>>(),
        );
    });
    let mut out = Array2::<Value>::default((data.len(), data[0].len()));
    out.axis_iter_mut(Axis(0))
        .enumerate()
        .for_each(|(idx1, mut row)| {
            row.iter_mut().enumerate().for_each(|(idx2, col)| {
                *col = data[idx1][idx2];
            });
        });
    (out, labels)
}
