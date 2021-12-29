use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::fs::File;
use std::io::{stdout, BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::str::FromStr;

use ndarray::{Array2, Axis};
use num_traits::Float;

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
    let reader = BufReader::new(File::open(p).expect("Unable to open file"));
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

pub(crate) fn display_results<L>(
    converged: bool,
    results: &HashMap<usize, Vec<usize>>,
    labels: &[L],
) where
    L: Display + AsRef<[u8]>,
{
    let mut writer = BufWriter::new(stdout());
    // Output header
    writer
        .write(
            format!(
                "Converged={} nClusters={} nSamples={}\n",
                converged,
                results.len(),
                results.iter().map(|(_, v)| v.len()).sum::<usize>()
            )
            .as_ref(),
        )
        .unwrap();
    results.iter().enumerate().for_each(|(idx, (key, value))| {
        // Write each exemplar
        writer
            .write(
                format!(
                    ">Cluster={} size={} exemplar={}\n",
                    idx + 1,
                    value.len(),
                    labels[*key]
                )
                .as_ref(),
            )
            .unwrap();
        // Write exemplar members
        let mut it = value.iter();
        writer.write(labels[*it.next().unwrap()].as_ref()).unwrap();
        it.for_each(|v| {
            writer.write(b",").unwrap();
            writer.write(labels[*v].as_ref()).unwrap();
        });
        writer.write(b"\n").unwrap();
    });
    writer.flush().unwrap();
}

#[cfg(test)]
mod test {
    use crate::from_file;
    use ndarray::arr2;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn valid_load() {
        // Write tempdata
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "{}\t{}\t{}\t{}", "id1", "1.0", "5.0", "1.0").unwrap();
        writeln!(file, "{}\t{}\t{}\t{}", "id2", "2.0", "4.0", "2.0").unwrap();
        writeln!(file, "{}\t{}\t{}\t{}", "id3", "3.0", "3.0", "3.0").unwrap();
        writeln!(file, "{}\t{}\t{}\t{}", "id4", "4.0", "2.0", "4.0").unwrap();
        writeln!(file, "{}\t{}\t{}\t{}", "id5", "5.0", "1.0", "5.0").unwrap();
        // Read into starting data
        let (data, labels) = from_file::<f32>(file.path().to_path_buf());
        // Validate ids
        for i in 0..5 {
            assert_eq!("id".to_string() + &(i + 1).to_string(), labels[i as usize]);
        }
        // Validate remaining
        let expected = arr2(&[
            [1., 5., 1.],
            [2., 4., 2.],
            [3., 3., 3.],
            [4., 2., 4.],
            [5., 1., 5.],
        ]);
        assert_eq!(data, expected);
    }

    #[test]
    #[should_panic]
    fn invalid_load_empty_file() {
        let file = NamedTempFile::new().unwrap();
        let (_, _) = from_file::<f32>(file.path().to_path_buf());
    }

    #[test]
    #[should_panic]
    fn invalid_load_mismatched_data() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "{}\t{}\t{}\t{}", "id1", "1.0", "5.0", "1.0").unwrap();
        writeln!(file, "{}\t{}\t{}", "id2", "2.0", "4.0").unwrap();
        writeln!(file, "{}\t{}\t{}\t{}", "id1", "1.0", "5.0", "1.0").unwrap();
        let (_, _) = from_file::<f32>(file.path().to_path_buf());
    }

    #[test]
    #[should_panic]
    fn invalid_load_invalid_data() {
        // Write tempdata
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "{}\t{}\t{}\t{}", "id1", "1.0", "5.0", "1.0").unwrap();
        writeln!(file, "{}\t{}\t{}\t{}", "id2", "a", "b", "c").unwrap();
        let (_, _) = from_file::<f32>(file.path().to_path_buf());
    }
}
