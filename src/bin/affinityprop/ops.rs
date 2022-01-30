use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::fs::File;
use std::io::{stdout, BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::str::FromStr;

use ndarray::{Array2, Axis};
use num_traits::Float;

#[derive(Debug)]
pub(crate) struct FileParseError {
    pub message: String,
}

/// Reads in a file formatted as (tab separated):
///     id1 val1 val2 val3
///     id2 val1 val2 val3
///
/// Provide as many ids and values as desired
/// All rows should be same length
/// Values should be floating-point decimal values
pub(crate) fn from_file<F>(
    p: PathBuf,
    d: &str,
    is_precalculated: bool,
) -> Result<(Array2<F>, Vec<String>), FileParseError>
where
    F: Float + Default + FromStr,
    <F as FromStr>::Err: Debug,
{
    let reader = BufReader::new(File::open(p).expect("Unable to open file"));
    let mut labels = Vec::new();
    let mut data = Vec::new();
    let mut label: usize = 0;
    // Read tab-delimited file
    for (idx, line) in reader.lines().map(|l| l.unwrap()).enumerate() {
        if !line.contains(d) {
            return Err(FileParseError {
                message: "Input file is not tab-delimited".to_string(),
            });
        }
        let mut line = line.split(d);
        // ID as first col if not precalculated
        if !is_precalculated {
            let id = match line.next() {
                Some(l) => l.to_string(),
                None => {
                    return Err(FileParseError {
                        message: "Error loading line label".to_string(),
                    })
                }
            };
            labels.push(id);
        } else {
            labels.push(label.to_string());
            label += 1;
        }
        let mut entry: Vec<F> = vec![];
        for s in line {
            match s.parse::<F>() {
                Ok(v) => {
                    entry.push(v);
                }
                Err(_) => {
                    return Err(FileParseError {
                        message: format!("Error parsing file at line {}", idx + 1),
                    })
                }
            };
        }
        // Rest are data
        data.push(entry);
    }
    // Validate data was loaded
    if data.len() <= 1 {
        return Err(FileParseError {
            message: "Data file is empty or only contains a single entry".to_string(),
        });
    }
    let length;
    let message;
    if is_precalculated {
        // Validate data all has same length
        length = data.len();
        message = "Precalculated input data must be square!".to_string();
    } else {
        // Validate data all has same length
        length = data[0].len();
        message = "Input data rows must all be same length!".to_string();
    }
    for v in data.iter() {
        if v.len() != length {
            return Err(FileParseError { message });
        }
    }
    // Convert data to Array2
    let mut out = Array2::<F>::default((data.len(), data[0].len()));
    out.axis_iter_mut(Axis(0))
        .enumerate()
        .for_each(|(idx1, mut row)| {
            row.iter_mut().enumerate().for_each(|(idx2, col)| {
                *col = data[idx1][idx2];
            });
        });
    Ok((out, labels))
}

#[cfg(not(tarpaulin_include))]
pub(crate) fn display_results<L>(
    converged: bool,
    results: &HashMap<usize, Vec<usize>>,
    labels: Vec<L>,
) where
    L: Display + AsRef<[u8]>,
{
    let mut writer = BufWriter::new(stdout());
    // Output header
    writer
        .write_all(
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
            .write_all(
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
        writer
            .write_all(labels[*it.next().unwrap()].as_ref())
            .unwrap();
        it.for_each(|v| {
            writer.write_all(b",").unwrap();
            writer.write_all(labels[*v].as_ref()).unwrap();
        });
        writer.write_all(b"\n").unwrap();
    });
    writer.flush().unwrap();
}

#[cfg(test)]
mod test {
    use std::io::Write;

    use ndarray::arr2;
    use tempfile::NamedTempFile;

    use crate::from_file;

    #[test]
    fn valid_load() {
        // Write tempdata
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "id1\t1.0\t5.0\t1.0").unwrap();
        writeln!(file, "id2\t2.0\t4.0\t2.0").unwrap();
        writeln!(file, "id3\t3.0\t3.0\t3.0").unwrap();
        writeln!(file, "id4\t4.0\t2.0\t4.0").unwrap();
        writeln!(file, "id5\t5.0\t1.0\t5.0").unwrap();
        // Read into starting data
        let (data, labels) = from_file::<f32>(file.path().to_path_buf(), "\t", false).unwrap();
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
        let (_, _) = from_file::<f32>(file.path().to_path_buf(), "\t", false).unwrap();
    }

    #[test]
    #[should_panic]
    fn invalid_load_mismatched_data() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "id1\t1.0\t5.0\t1.0").unwrap();
        writeln!(file, "id2\t2.0\t4.0").unwrap();
        writeln!(file, "id3\t1.0\t5.0\t1.0").unwrap();
        let (_, _) = from_file::<f32>(file.path().to_path_buf(), "\t", false).unwrap();
    }

    #[test]
    #[should_panic]
    fn invalid_blank_line() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "id1\t1.0\t5.0\t1.0").unwrap();
        writeln!(file).unwrap();
        writeln!(file, "id3\t1.0\t5.0\t1.0").unwrap();
        let (_, _) = from_file::<f32>(file.path().to_path_buf(), "\t", false).unwrap();
    }

    #[test]
    #[should_panic]
    fn invalid_load_invalid_data() {
        // Write tempdata
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "id1\t1.0\t5.0\t1.0").unwrap();
        writeln!(file, "id2\ta\tb\tc").unwrap();
        let (_, _) = from_file::<f32>(file.path().to_path_buf(), "\t", false).unwrap();
    }

    #[test]
    #[should_panic]
    fn invalid_file_format() {
        // Write tempdata
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "id1 1.0 5.0 1.0").unwrap();
        writeln!(file, "id2 1.0 2.0 1.0").unwrap();
        let (_, _) = from_file::<f32>(file.path().to_path_buf(), "\t", false).unwrap();
    }

    #[test]
    fn precalculated_file_format() {
        // Write tempdata
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "0.0 -3.0 -12.0").unwrap();
        writeln!(file, "-3.0 0.0 -3.0").unwrap();
        writeln!(file, "-12.0 -3.0 0.0").unwrap();
        let (_, y) = from_file::<f32>(file.path().to_path_buf(), " ", true).unwrap();
        let mut expected_id: usize = 0;
        for id in y {
            assert_eq!(expected_id.to_string(), id);
            expected_id += 1;
        }
    }

    #[test]
    #[should_panic]
    fn invalid_precalculated_file_format() {
        // Write tempdata
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "0.0 -3.0 -12.0").unwrap();
        writeln!(file, "-12.0 -3.0 0.0").unwrap();
        let (_, _) = from_file::<f32>(file.path().to_path_buf(), " ", true).unwrap();
    }
}
