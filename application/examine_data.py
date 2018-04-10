#! /usr/bin/env python3
""" Standalone script for evaluating a dataset.
Calculates measures of label quality and tries to spot outliers.

Args:
    -f, --file the file to read """

import sys
import argparse
import pandas as pd


def main():
    """The executable to read in the specified data file and perform the 
    investigatory operations."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "input_file",type=argparse.FileType("r"), 
            help="The path of the file to read in.")
    args = parser.parse_args()

    #read the dataframe in memory friendly chunks
    reader = pd.read_csv(args.input_file, iterator=True, chunksize=)
    df = pd.concat(reader, ignore_index=True)

if __name__ == "__main__":
    main()
