#! /usr/bin/env python3
""" Standalone script for evaluating a dataset.
Calculates measures of label quality and tries to spot outliers.

Usage:
    python3 investigate.py <filename>"""

import argparse
import os.path
from collections import defaultdict
import pandas as pd

class Investigator(object):
    """Utility class for calculating statistics on a data set."""

    def __init__(self):
        """Initializes basic statistics."""
        self.entry_count = 0
        self.artists = set([])
        self.genres = defaultdict(int)
        self.word_counts = defaultdict(int)
        self.year_counts = defaultdict(int)
        self.durations = []

    def process_row(self, data, label):
        """Procceses a pandas series obect."""
        self.entry_count += 1
        self.artists.add(data["artist_name"])
        self.genres[label["genre"]] += 1
        count = Investigator.__count_words(data["lyrics"])
        self.word_counts[count] += 1
        self.year_counts[data["year"]] += 1
        self.durations.append(data["duration"])

    @staticmethod
    def __count_words(bow):
        """counts the unique words in a bag of words format"""
        if isinstance(bow, str):
            return len(bow.split())
        return 0

    def generate_statistics(self):
        """Gives a dict of the most important statistics of the dataset."""
        output = pd.Series()
        output["entry_count"] = self.entry_count
        output["unique_artists"] = len(self.artists)
        output["genre_counts"] = dict(self.genres)
        output["year_counts"] = sorted(
            self.year_counts.items(),
            key=lambda k_v: k_v[1])
        output["word_frequencies"] = dict(self.word_counts)
        output["duration"] = sorted(self.durations)
        print(output)

        return output

def main():
    """The executable to read in the specified data file and perform the
    investigatory operations."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        type=argparse.FileType("r"),
        help="The path of the data file to read in.")
    parser.add_argument(
        "input_label_file",
        type=argparse.FileType("r"),
        help="The path of the label file to read in.")
    parser.add_argument(
        "output_path",
        default="",
        help="The path to write to.")

    args = parser.parse_args()

    #read the dataframe in memory friendly chunks
    data_frame = pd.read_pickle(args.input_file.name)
    label_frame = pd.read_pickle(args.input_label_file.name)

    investigator = Investigator()

    for series in zip(data_frame.iterrows(), label_frame.iterrows()):
        investigator.process_row(series[0][1], series[1][1])

    series = investigator.generate_statistics()


    filename = os.path.basename(args.input_file.name)[:-4] + "_stats.pkl"
    series.to_pickle(os.path.join(args.output_path, filename))

if __name__ == "__main__":
    main()
