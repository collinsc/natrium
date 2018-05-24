#! /usr/bin/env python3
""" Standalone script for evaluating a dataset.
Calculates measures of label quality and tries to spot outliers.

Usage:
    python3 investigate.py <filename>"""

import argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def center_and_scale(frame):
    """Centers and scales a pandas dataframe"""
    for key, col in frame.iteritems():
        if pd.api.types.is_numeric_dtype(col):
            mean = col.mean()
            dev = col.std()
            col = (mean - col)/dev

def calculate_scree(S):
    """Calculates information for scree plot"""
    S2= S**2
    S2 = S2/sum(S2)
    return S2, np.cumsum(S2)

def plot_scree(scores, cum_scores):
    x = np.linspace(1, scores.shape[0], scores.shape[0])

    fig, ax = plt.subplots()

    ax.plot(x, scores,label="variance")
    ax.plot(x, cum_scores, label="cumulative_variance")
    ax.set(xlabel='Principal Component', ylabel='Variance',
           title='Scree Plots')
    ax.legend()
    plt.show()
    
        


def main():
    """The executable to read in the specified data file and perform the
    investigatory operations."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        type=argparse.FileType("r"),
        help="The path of the data file to read in.")

    args = parser.parse_args()

    #read the dataframe in memory friendly chunks
    data_frame = pd.read_pickle(args.input_file.name).infer_objects()
    center_and_scale(data_frame)
    labels = data_frame["genre"]

    data_frame = data_frame.drop(columns=["genre", "release_year"])

    u, s, v = np.linalg.svd(data_frame, full_matrices = False)

    scores, cumulative_scores = calculate_scree(s)
    plot_scree(scores, cumulative_scores)

    


if __name__ == "__main__":
    main()
