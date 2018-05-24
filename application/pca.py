#! /usr/bin/env python3
""" Standalone script for evaluating a dataset.
Calculates measures of label quality and tries to spot outliers.

Usage:
    python3 investigate.py <filename>"""

import argparse
import pandas as pd
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
    """Displays the scree and cumulative scree plot in the same figure"""
    x = np.linspace(1, scores.shape[0], scores.shape[0])

    fig, ax = plt.subplots()

    ax.plot(x, scores,label="variance")
    ax.plot(x, cum_scores, label="cumulative_variance")
    ax.set(xlabel='Principal Component', ylabel='Variance',
           title='Scree Plots')
    ax.grid(True)
    ax.legend()

def calculate_loading(V):
    return np.divide(V**2,np.sign(V))

def plot_loading_vectors(V2):
    size = math.ceil(math.sqrt(V2.shape[0]))
    fig, axes = plt.subplots(size, size, figsize=(14,8))
    vectors = range(1,15)
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            idx = i*size+j
            if (idx < V2.shape[0]):
                ax.bar(vectors, V2[idx],tick_label=vectors)
                y_lim = np.max(np.abs(V2[idx])) + 0.1
                ax.set_ylim([-y_lim, y_lim])
                ax.set(title="Vector %d" % (idx + 1), xlabel="Original Dimension", ylabel="Importance" )
                plt.setp(ax.get_xticklabels(), rotation=45)
                ax.axhline(lw = 0.5,color="black")
            else:
                ax.axis("off")
    fig.suptitle("Loading Vector Plots", size = 18)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

def plot_features_3d(size, data, labels, cols, axes_labels, series_names, series_label, title):
    """makes a 3d plot of a set of data"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    sample = labels.sample(n=size)
    for label, series in sample.groupby(sample):
        idx = series.index
        ax.scatter(data.ix[idx][cols[0]], data.ix[idx][cols[1]], data.ix[idx][cols[2]], depthshade=False, label = label)

    ax.set_xlabel(axes_labels[0])
    ax.set_ylabel(axes_labels[1])
    ax.set_zlabel(axes_labels[2])
    ax.set_title(title, y=1.08)
    ax.legend()



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
    genre_list = ["Punk", "Electronic","RnB", "Rap", "Country", "Metal", "Pop", "Rock"]
    data_frame = data_frame[data_frame["genre"].isin(genre_list)]
    center_and_scale(data_frame)
    labels = data_frame["genre"]

    data_frame = data_frame.drop(columns=["genre", "release_year"])

    u, s, v = np.linalg.svd(data_frame, full_matrices = False)
    ur = np.dot(u, np.diag(s))
    regular_scores = pd.DataFrame(data=ur, 
            index = data_frame.index,
            columns = range(1, ur.shape[1]+1))

    scores, cumulative_scores = calculate_scree(s)
    plot_scree(scores, cumulative_scores)

    v2 = calculate_loading(v)
    plot_loading_vectors(v2)

    plot_features_3d(
            150, 
            data_frame, 
            labels,
            ["duration", "word_count", "rhyme_value"], 
            ["Duration (s)", "Words (ct)", "Rhyme Index"],
            genre_list, 
            "genre",
            "Duration v. Wordcound v. Rhyme Index")

    plot_features_3d(
            150, 
            regular_scores, 
            labels,
            [1, 2, 3], 
            ["", "", ""],
            genre_list, 
            "genre",
            "Vector 1 v. Vector 2 v. Vector 3")

    plt.show()

if __name__ == "__main__":
    main()
