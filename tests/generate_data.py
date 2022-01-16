#!/usr/bin/env python
"""
Testing data generation script

Installation:
pip install scikit-learn

Tested using scikit-learn==0.24.2 on python 3.6.13

Usage:
./generate_data

Output:

"""
import os
import shutil

from sklearn.datasets import make_blobs


def generate(file_id: str, num_clusters: int, num_samples: int, dim: int, cluster_std: float):
    with open(f"data/{file_id}.test", "w") as test_file:
        for point, label in zip(*make_blobs(n_samples=num_samples, centers=num_clusters, n_features=dim, random_state=0,
                                            cluster_std=cluster_std)):
            test_file.write(str(label))
            for p in point:
                test_file.write(" ")
                test_file.write(str(p))
            test_file.write("\n")


if __name__ == "__main__":
    if os.path.exists("data"):
        shutil.rmtree("data")
    os.makedirs("data")

    generate("near-exemplar-10", 10, 1000, 50, 1.0)
    generate("near-exemplar-100", 100, 1000, 50, 1.0)

