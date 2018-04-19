#!/usr/bin/env python3
"""Tool for visulizing the genres of a data series
see -h for help"""


import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        type=argparse.FileType("r"),
        help="The path of the file to read in.")
    args = parser.parse_args()


    series = pd.read_pickle(args.input_file.name)
    print(series)

if __name__ == "__main__":
    main()
