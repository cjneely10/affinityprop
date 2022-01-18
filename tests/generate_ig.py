#!/usr/bin/env python
"""
Generate `binsanity.test` file from BinSanity example data

Tested using Python 3.6.13


Usage:
./generate_ig
"""
import os
from pathlib import Path

# Directory of results from BinSanity example
from typing import Dict

FASTA_RESULTS_DIR = os.environ.get("BINSANITY_RESULTS_DIR")
if FASTA_RESULTS_DIR is None:
    exit("BINSANITY_RESULTS_DIR is not set")

# Write directory
DATA_DIR = "test-data"


def get_ids_from_fasta(fasta_file: Path) -> set:
    ids = set()
    with open(fasta_file, "r") as file_ptr:
        for line in file_ptr:
            if line.startswith(">"):
                ids.add(line.split(" ")[0][1:].rstrip("\r\n"))
    return ids


def get_data_from_ig_file(ig_file: Path) -> Dict[str, str]:
    _point_data = {}
    with open(ig_file, "r") as ig_ptr:
        for line in ig_ptr:
            data = line.split("\t")
            _point_data[data[0]] = " ".join(data[1:])
    return _point_data


if __name__ == "__main__":
    with open(f"{DATA_DIR}/binsanity.test", "w") as out_ptr:
        cluster_id = 0
        point_data = get_data_from_ig_file(Path(DATA_DIR).joinpath("Infant_gut_assembly.cov.x100.lognorm"))
        for file in os.listdir(FASTA_RESULTS_DIR):
            if file.endswith(".fna"):
                file = Path(os.path.join(FASTA_RESULTS_DIR, file)).resolve()
                for _id in get_ids_from_fasta(file):
                    out_ptr.write(f"{cluster_id} {point_data[_id]}")
                cluster_id += 1
