#! /usr/bin/env python3
""" Standalone script for evaluating a dataset.
Calculates measures of label quality and tries to spot outliers.

Args:
    -f, --file the file to read """

import sys
import argparse
import pandas as pd

class Investigator(object):
    """Utility class for calculating statistics on a data set."""

    def __init__(self):
        """Initializes basic statistics."""
        self.entry_count = 0

    def proccess_row(self, row):
        """Proccesses a pandas series obect."""
        self.entry_count += 1




def main():
    """The executable to read in the specified data file and perform the 
    investigatory operations."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "input_file",type=argparse.FileType("r"), 
            help="The path of the file to read in.")
    args = parser.parse_args()

    #read the dataframe in memory friendly chunks
    reader = pd.read_csv(args.input_file, iterator=True, chunksize=512)
    df = pd.concat(reader, ignore_index=True)

    investigator = Investigator()

    for series in df.iterrows():
        investigator.proccess_row(series)

    print(investigator.__dict__)

if __name__ == "__main__":
    main()
