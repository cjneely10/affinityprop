use crate::affinity_propagation::Value;
use ndarray::{Array2, Axis};
use std::fmt::Debug;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::str::FromStr;

/// Reads in a file formatted as (single-space separated):
///     id1 val1 val2 val3
///     id2 val1 val2 val3
///
/// Provide as many ids and values as desired
/// All rows should be same length
/// Values should be floating-point decimal values
pub fn from_file<L>(p: PathBuf) -> (Array2<Value>, Vec<L>)
where
    L: FromStr,
    <L as FromStr>::Err: Debug,
{
    let reader = BufReader::new(File::open(p).unwrap());
    let mut labels = Vec::new();
    let mut data = Vec::new();
    reader.lines().map(|l| l.unwrap()).for_each(|line| {
        let mut line = line.split(' ');
        labels.push(line.next().unwrap().parse::<L>().unwrap());
        data.push(
            line.map(|s| s.parse::<Value>().unwrap())
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
