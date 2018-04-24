#!/usr/bin/env python3
"""Tool for visualizing the lyrics breakdown.
See -h for help"""

import argparse
import pandas as pd
import pandas.core.frame
import sqlite3
import matplotlib as mpl
mpl.use('Agg')  # Necessary so we can work without Tkinter
import matplotlib.pyplot as plt
import numpy as np


def generate_hist(data: list, filename: str):
    plt.close()
    axis_nums = np.arange(len(data))
    plt.figure(figsize=(11, 7))
    plt.bar(axis_nums, [x[1] for x in data], align='center', width=1.0)
    plt.title("Word Counts (Top 50 Words)", fontsize=18)
    plt.xlabel("Word")
    plt.ylabel("Count", fontsize=14)
    plt.xticks(axis_nums, [x[0] for x in data], fontsize=6)
    plt.xticks(rotation=75)
    plt.yticks(fontsize=10)
    plt.savefig(filename, dpi=480)


def main():
    """Generates various visualizations for lyrics"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        type=argparse.FileType("r"),
        help="The path of the file to read in.")
    args = parser.parse_args()

    series = pd.read_pickle(args.input_file.name)
    assert isinstance(series, pd.core.frame.DataFrame)

    lyric_freqs = {}

    # Process the lyrics and store them in a dictionary
    lyrics = list(series.iloc[:, 7])

    for song in lyrics:
        for lyric in song.split(' '):
            pair = lyric.split(':')

            if pair[0] not in lyric_freqs:
                lyric_freqs[int(pair[1])] = int(pair[0])
            else:
                lyric_freqs[int(pair[1])] += int(pair[0])

    # Sort the lyrics by frequency
    lyric_freq_sorted = []
    for lyric in lyric_freqs.items():
        lyric_freq_sorted.append((lyric[0], lyric[1]))

    lyric_freq_sorted.sort(key=lambda x: x[1])

    # Just process the top 50
    lyric_freq_sorted = lyric_freq_sorted[-50:]

    # Map the word indices to the actual written word
    lyric_freq_sorted_mapped = []
    with sqlite3.connect('musixmatch_lyrics.db') as lyrics_db:
        words_c = lyrics_db.cursor()
        for entry in lyric_freq_sorted:
            words_c.execute("SELECT word FROM words_indexed WHERE number LIKE {}".format(entry[0]))
            lyric_freq_sorted_mapped.append((words_c.fetchone()[0], entry[1]))

    for pair in lyric_freq_sorted_mapped:
        print("{}: {} occurrences".format(pair[0], pair[1]))

    # Make a graph!
    generate_hist(lyric_freq_sorted_mapped, 'visualizations/word_hist_byfrequency.png')


if __name__ == "__main__":
    main()