#! /usr/bin/env python3
"""Script for partitioning groomed data set into a training and testing

Usage:
    python3 remove_columns.py <input_file> <output_file> <list of space delimited collumn labels>
"""

import argparse
import pandas as pd
import numpy as np

def ratio_check(val):
    """Gives if a floating point number is a valid ratio"""
    f_val = float(val)
    if f_val < 0 or f_val > 1:
        raise argparse.ArgumentTypeError("Ratio must be between 0 and 1")
    return f_val

def main():
    """Partitions the data based on user ratio."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_file",
        type=argparse.FileType("r"),
        help="The path of the file to read in.")

    parser.add_argument(
        "ratio",
        type=float,
        help="The ratio of how to much of the data we want to train with.")

    parser.add_argument(
        "--training_file",
        type=argparse.FileType("w"),
        default="train.csv",
        help="The path of the training file to write to.")

    parser.add_argument(
        "--testing_file",
        type=argparse.FileType("w"),
        default="test.csv",
        help="The path of the testing file to write to.")


    args = parser.parse_args()

    #read the dataframe in memory friendly chunks
    data_frame = pd.read_pickle(args.input_file.name)

    #create neccessary data structures
    print("Columns in Dataset:", list(data_frame))
    entry_num = data_frame.shape[0]
    print("Number of entries:", entry_num)
    print("Ratio:", args.ratio)
    train = pd.DataFrame(columns=data_frame.columns)
    test = pd.DataFrame(columns=data_frame.columns)

    #partition the data
    data_frame.loc[np.random.permutation(data_frame.index)]
    partition_idx = int(entry_num*args.ratio)
    train, test = np.split(data_frame, [partition_idx])

    train.to_pickle(args.training_file.name)
    test.to_pickle(args.testing_file.name)

    print("Success")
    print("Training entries:", train.shape[0])
    print("Testing entries:", test.shape[0])

if __name__ == "__main__":
    main()
