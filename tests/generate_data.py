#!/usr/bin/env python
"""
Testing data generation script

Installation (tested using Python 3.6.13):
pip install scikit-learn==0.24.2

Usage:
./generate_data
"""
import os
import shutil

from sklearn.datasets import make_blobs

DATA_DIR = "test-data"


def generate(num_clusters: int, num_samples: int, dim: int, cluster_std: float):
    with open(f"{DATA_DIR}/near-exemplar-{num_clusters}.test", "w") as test_file:
        for point, label in zip(*make_blobs(n_samples=num_samples, centers=num_clusters, n_features=dim, random_state=0,
                                            cluster_std=cluster_std)):
            # label [data...]
            test_file.write(str(label))
            for p in point:
                test_file.write(" ")
                test_file.write(str(p))
            test_file.write("\n")


if __name__ == "__main__":
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR)

    # 10 clusters, 100 data points
    generate(10, 300, 50, 1.0)
    generate(50, 300, 50, 1.0)

