#! /usr/bin/env python3
"""
Standalone script for evaluating a dataset.

Calculates measures of label quality and tries to spot outliers.

Args:
    -f, --file the file to read
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file",help="The file to read in.")
    args = parser.parse_args()

if __name__ == "__main__":
    main()
