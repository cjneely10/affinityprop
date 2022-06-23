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

/// Reads in a file formatted as (delimiter-separated):
///     id1 val1 val2 val3
///     id2 val1 val2 val3
///
/// Provide as many ids and values as desired
/// All rows should be same length
/// Values should be floating-point decimal values
pub(crate) fn from_file<F>(
    p: PathBuf,
    d: &str,
    collect_labels: bool,
    is_precalculated: bool,
) -> Result<(Array2<F>, Vec<String>), FileParseError>
where
    F: Float + Default + FromStr,
    <F as FromStr>::Err: Debug + Display,
{
    let reader = BufReader::new(File::open(p).expect("Unable to open file"));
    let mut labels = Vec::new();
    let mut data = Vec::new();
    let mut label: usize = 0;
    // Read tab-delimited file
    for (idx, line) in reader.lines().map(|l| l.unwrap()).enumerate() {
        if line.is_empty() {
            return Err(FileParseError {
                message: format!("line {}: empty line detected", idx + 1),
            });
        }
        if !line.contains(d) {
            return Err(FileParseError {
                message: format!("line {}: not properly delimited", idx + 1),
            });
        }
        let mut line = line.split(d);
        // ID as first col if not precalculated
        if collect_labels {
            if let Some(l) = line.next() {
                if !l.is_empty() {
                    labels.push(l.to_string());
                } else {
                    return Err(FileParseError {
                        message: format!("line {}: empty line label `{}`", idx + 1, l),
                    });
                }
            }
        } else {
            labels.push(label.to_string());
            label += 1;
        }
        let mut entry: Vec<F> = vec![];
        for (i, s) in line.enumerate() {
            match s.parse::<F>() {
                Ok(v) => {
                    if v.is_nan() {
                        return Err(FileParseError {
                            message: format!("line {} col {}: nan value detected", idx + 1, i + 1),
                        });
                    }
                    entry.push(v);
                }
                Err(_) => {
                    return Err(FileParseError {
                        message: format!(
                            "line {} col {}: unable to convert `{}` to float",
                            idx + 1,
                            i + 1,
                            s
                        ),
                    });
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
        message = "Precalculated input data must be square";
    } else {
        // Validate data all has same length
        length = data[0].len();
        message = "Input data rows must all be same length";
    }
    for (i, v) in data.iter().enumerate() {
        if v.len() != length {
            return Err(FileParseError {
                message: format!(
                    "Error at line {}: {}\n\tlength = {}\n\texpected = {}",
                    i + 1,
                    message,
                    v.len(),
                    length
                ),
            });
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
    write_header(&mut writer, results, converged);
    // Write exemplar members
    results.iter().enumerate().for_each(|(idx, (&key, value))| {
        write_exemplar(&mut writer, idx + 1, key, value, &labels);
    });
    writer.flush().unwrap();
}

#[cfg(not(tarpaulin_include))]
fn write_header<W: Write>(writer: &mut W, results: &HashMap<usize, Vec<usize>>, converged: bool) {
    writer.write_all(b"Converged=").unwrap();
    writer.write_all(converged.to_string().as_bytes()).unwrap();
    writer.write_all(b" nClusters=").unwrap();
    writer
        .write_all(results.len().to_string().as_bytes())
        .unwrap();
    writer.write_all(b" nSamples=").unwrap();
    writer
        .write_all(
            results
                .iter()
                .map(|(_, v)| v.len())
                .sum::<usize>()
                .to_string()
                .as_bytes(),
        )
        .unwrap();
    writer.write_all(b"\n").unwrap();
}

#[cfg(not(tarpaulin_include))]
fn write_exemplar<W, L>(
    writer: &mut W,
    cluster_idx: usize,
    key: usize,
    value: &[usize],
    labels: &[L],
) where
    W: Write,
    L: Display + AsRef<[u8]>,
{
    writer.write_all(b">Cluster=").unwrap();
    writer
        .write_all(cluster_idx.to_string().as_bytes())
        .unwrap();
    writer.write_all(b" size=").unwrap();
    writer
        .write_all(value.len().to_string().as_bytes())
        .unwrap();
    writer.write_all(b" exemplar=").unwrap();
    writer
        .write_all(labels[key].to_string().as_bytes())
        .unwrap();
    writer.write_all(b"\n").unwrap();

    let mut it = value.iter().copied();
    writer
        .write_all(labels[it.next().unwrap()].as_ref())
        .unwrap();
    it.for_each(|v| {
        writer.write_all(b",").unwrap();
        writer.write_all(labels[v].as_ref()).unwrap();
    });
    writer.write_all(b"\n").unwrap();
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
        let (data, labels) =
            from_file::<f32>(file.path().to_path_buf(), "\t", true, false).unwrap();
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
        let (_, _) = from_file::<f32>(file.path().to_path_buf(), "\t", true, false).unwrap();
    }

    #[test]
    #[should_panic]
    fn invalid_contains_nan() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "id1\t1.0\t5.0\t1.0").unwrap();
        writeln!(file, "id2\t2.0\t4.0\tnan").unwrap();
        let (_, _) = from_file::<f32>(file.path().to_path_buf(), "\t", true, false).unwrap();
    }

    #[test]
    #[should_panic]
    fn invalid_misformatted_file() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "ID1\t1.0\t2.0\t3.0").unwrap();
        writeln!(file, "\t1.0\t2.0\t3.0").unwrap();
        let (_, _) = from_file::<f32>(file.path().to_path_buf(), "\t", true, false).unwrap();
    }

    #[test]
    #[should_panic]
    fn invalid_load_mismatched_data() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "id1\t1.0\t5.0\t1.0").unwrap();
        writeln!(file, "id2\t2.0\t4.0").unwrap();
        writeln!(file, "id3\t1.0\t5.0\t1.0").unwrap();
        let (_, _) = from_file::<f32>(file.path().to_path_buf(), "\t", true, false).unwrap();
    }

    #[test]
    #[should_panic]
    fn invalid_blank_line() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "id1\t1.0\t5.0\t1.0").unwrap();
        writeln!(file, "").unwrap();
        writeln!(file, "id3\t1.0\t5.0\t1.0").unwrap();
        let (_, _) = from_file::<f32>(file.path().to_path_buf(), "\t", true, false).unwrap();
    }

    #[test]
    #[should_panic]
    fn invalid_load_invalid_data() {
        // Write tempdata
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "id1\t1.0\t5.0\t1.0").unwrap();
        writeln!(file, "id2\ta\tb\tc").unwrap();
        let (_, _) = from_file::<f32>(file.path().to_path_buf(), "\t", true, false).unwrap();
    }

    #[test]
    #[should_panic]
    fn invalid_file_format() {
        // Write tempdata
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "id1 1.0 5.0 1.0").unwrap();
        writeln!(file, "id2 1.0 2.0 1.0").unwrap();
        let (_, _) = from_file::<f32>(file.path().to_path_buf(), "\t", true, false).unwrap();
    }

    #[test]
    fn precalculated_file_format() {
        // Write tempdata
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "0.0 -3.0 -12.0").unwrap();
        writeln!(file, "-3.0 0.0 -3.0").unwrap();
        writeln!(file, "-12.0 -3.0 0.0").unwrap();
        let (_, y) = from_file::<f32>(file.path().to_path_buf(), " ", false, true).unwrap();
        let mut expected_id: usize = 0;
        for id in y {
            assert_eq!(expected_id.to_string(), id);
            expected_id += 1;
        }
    }

    #[test]
    fn precalculated_file_format_with_labels() {
        // Write tempdata
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "ID1 0.0 -3.0 -12.0").unwrap();
        writeln!(file, "ID2 -3.0 0.0 -3.0").unwrap();
        writeln!(file, "ID3 -12.0 -3.0 0.0").unwrap();
        let (_, y) = from_file::<f32>(file.path().to_path_buf(), " ", true, true).unwrap();
        let mut expected_id: usize = 1;
        for id in y {
            assert_eq!("ID".to_string() + &expected_id.to_string(), id);
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
        let (_, _) = from_file::<f32>(file.path().to_path_buf(), " ", false, true).unwrap();
    }
}
