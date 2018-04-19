#! /usr/bin/env python3
"""Script for partitioning groomed data set into a training and testing 

Usage:
    python3 remove_columns.py <input_file> <output_file> <list of space delimited collumn labels>
"""

import pandas as pd
import numpy as np
import argparse
import random

def ratio_check(val):
    """Gives if a floating point number is a valid ratio"""
    f_val = float(val)
    if f_val < 0 or f_val > 1:
        raise argparse.ArgumentTypeError("Ratio must be between 0 and 1")
    return f_val

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
           "input_file",
           type=argparse.FileType("r"),
           help="The path of the file to read in.")

    parser.add_argument("ratio",
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
    reader = pd.read_csv(args.input_file, iterator=True, chunksize=512)
    data_frame = pd.concat(reader, ignore_index=True)

    #create neccessary data structures
    print("Columns in Dataset:", list(data_frame))
    entry_num =data_frame.shape[0]
    print("Number of entries:", entry_num)
    print("Ratio:", args.ratio)
    train = pd.DataFrame(columns=data_frame.columns)
    test = pd.DataFrame(columns=data_frame.columns)

    #partition the data
    data_frame.loc[np.random.permutation(data_frame.index)]
    partition_idx = int(entry_num*args.ratio)
    train, test = np.split(data_frame, [partition_idx])

    train.to_csv(args.training_file)
    test.to_csv(args.testing_file)

    print("Success")
    print("Training entries:", train.shape[0])
    print("Testing entries:", test.shape[0])

if __name__ == "__main__":
    main()
