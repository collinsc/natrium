#!/usr/bin/env python3
"""Tool for visualizing the part-of-speech distributions of a data series.
See -h for help"""

import argparse
import pandas as pd
import pandas.core.frame
import matplotlib as mpl
mpl.use('Agg')  # Necessary so we can work without Tkinter
import matplotlib.pyplot as plt
import numpy as np
import json


def main():
    """Generates a graph of the breakdown of parts-of-speech"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        type=argparse.FileType("r"),
        help="The path of the file to read in.")
    args = parser.parse_args()

    genre_data = json.loads(open(args.input_file.name).read())
    print(genre_data.keys())


if __name__ == "__main__":
    main()