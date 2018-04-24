#! /usr/bin/env python3
"""Script for removing columns from a data set.

Output data stored as a pkl file.

Usage:
    python3 remove_columns.py <input_file> <output_file> <list of space delimited collumn labels>
"""

import pandas as pd
import argparse

def main():
    """Removes user defined columns stores in pkl file"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_file",
        type=argparse.FileType("r"),
        help="The path of the file to read in.")

    parser.add_argument(
        "output_file",
        type=argparse.FileType("w"),
        help="The path of the file to write to.")

    parser.add_argument(
        "column_labels",
        nargs="+",
        type=str,
        help="The list of column labels to exclude")
    args = parser.parse_args()

    #read the dataframe in memory friendly chunks
    reader = pd.read_csv(args.input_file, iterator=True, chunksize=512)
    data_frame = pd.concat(reader, ignore_index=True)

    print("Columns in Dataset", list(data_frame))
    print("Removing Collumns: ", args.column_labels)
    data_frame.drop(columns=args.column_labels).to_pickle(args.output_file.name)
    print("removed")

if __name__ == "__main__":
    main()
