#!/usr/bin/env python3
"""Tool for visualizing the genres of a data series.
See -h for help"""

import argparse
import pandas as pd
import pandas.core.frame
import matplotlib as mpl
mpl.use('Agg')  # Necessary so we can work without Tkinter
import matplotlib.pyplot as plt
import numpy as np


def generate_hist(data: list, filename: str):
    plt.close()
    axis_nums = np.arange(len(data))
    plt.figure(figsize=(11, 11))
    plt.bar(axis_nums, [x[1] for x in data], align='center', width=1.0)
    plt.title("Genre Distribution", fontsize=24)
    plt.xlabel("Genre", fontsize=18)
    plt.ylabel("Count", fontsize=18)
    plt.xticks(axis_nums, [x[0] for x in data], fontsize=12)
    plt.xticks(rotation=75)
    plt.yticks(fontsize=10)
    plt.savefig(filename, dpi=480)


def generate_pie(data: list, filename: str):
    # Generate percentages off of original count data
    total_count = sum([x[1] for x in data])
    pie_values = [(x[0], x[1] / total_count) for x in data]

    plt.close()
    plt.pie([x[1] for x in pie_values], startangle=90, labels=[x[0] for x in pie_values], rotatelabels=45)
    # plt.legend(labels=[x[0] for x in pie_values], loc='best')
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

    # Gets the count of each genre
    counts = series.groupby('genre').genre.count()
    assert isinstance(counts, pd.core.frame.Series)

    # Translate to a list of tuples
    sorted_dict = []
    for c in counts.keys():
        sorted_dict.append((c, counts[c]))

    # Sort by genre for consistent plotting
    sorted_dict.sort(key=lambda x: x[0])

    # Make the plot and save it
    generate_hist(sorted_dict, 'visualizations/genre_hist_bygenre.png')

    # Re-sort by count for consistent plotting
    sorted_dict.sort(key=lambda x: x[1])

    # Make the plot and save it
    generate_hist(sorted_dict, 'visualizations/genre_hist_bycount.png')

    # Make a pie plot
    generate_pie(sorted_dict, 'visualizations/genre_pie.png')


if __name__ == "__main__":
    main()
