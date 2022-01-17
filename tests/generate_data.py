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

import sklearn.datasets as datasets

DATA_DIR = "test-data"


def prepared_x_y(fxn):
    def decorated(*args, **kwargs):
        return fxn(return_X_y=True, *args, **kwargs)
    return decorated


test_datasets = {
    "iris": prepared_x_y(datasets.load_iris),
    "diabetes": prepared_x_y(datasets.load_diabetes),
    "breast_cancer": prepared_x_y(datasets.load_breast_cancer)
}


def write(test_file, label, point):
    # label [data...]
    test_file.write(str(int(label)))
    for p in point:
        test_file.write(" ")
        test_file.write(str(p))
    test_file.write("\n")


def generate_gaussian(num_clusters: int, num_samples: int, dim: int, cluster_std: float):
    with open(f"{DATA_DIR}/near-exemplar-{num_clusters}.test", "w") as test_file:
        for point, label in zip(*datasets.make_blobs(n_samples=num_samples, centers=num_clusters, n_features=dim,
                                                     random_state=0, cluster_std=cluster_std)):
            write(test_file, label, point)


def generate_prepared():
    for key in test_datasets.keys():
        with open(f"{DATA_DIR}/{key}.test", "w") as test_file:
            for point, label in zip(*test_datasets[key]()):
                write(test_file, label, point)


if __name__ == "__main__":
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR)

    # 10 clusters, 100 data points
    generate_gaussian(10, 300, 50, 1.0)
    generate_gaussian(50, 300, 50, 1.0)
    generate_prepared()
