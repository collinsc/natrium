#!/usr/bin/env python3
"""Tool for visualizing the durations of a data series.
See -h for help"""

import argparse
import pandas as pd
import pandas.core.frame
import matplotlib as mpl
mpl.use('Agg')  # Necessary so we can work without Tkinter
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def generate_hist(data: list, binsize: int, filename: str):
    plt.close()
    plt.figure(figsize=(11, 7))
    plt.hist(data, bins=list(range(0, 800, binsize)))
    plt.title("Song Duration", fontsize=18)
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Count", fontsize=14)
    plt.xticks([x + (binsize / 2) for x in range(0, 800 - binsize, binsize)], rotation=75)
    plt.yticks(fontsize=10)
    plt.savefig(filename, dpi=480)


def generate_norm(data: list, filename: str):
    plt.close()
    plt.figure(figsize=(11, 7))
    plt.title("Song Duration", fontsize=18)
    fit = norm.pdf(sorted(data), np.mean(data), np.std(data))
    plt.plot(sorted(data), fit, '-')
    plt.plot(data, [0] * len(data), 'o')
    plt.savefig(filename, dpi=480)


def main():
    """Generates a graph of the breakdown of genres"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        type=argparse.FileType("r"),
        help="The path of the file to read in.")
    args = parser.parse_args()

    series = pd.read_pickle(args.input_file.name)
    assert isinstance(series, pd.core.frame.DataFrame)

    durations = list(series.iloc[:, 4])

    for i in range(20, 51, 10):
        generate_hist(durations, i, 'visualizations/duration_hist_binsize_{}.png'.format(i))

    generate_norm(durations, 'visualizations/duration_norm.png')


if __name__ == '__main__':
    main()
