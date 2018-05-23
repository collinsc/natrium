#!/usr/bin/env python3
"""Tool for generating correlation and coefficient matrices of the data."""

import numpy as np
import pandas
import json


def main():
    orig_data = pandas.read_pickle('data/all.pkl')[['track_id', 'duration', 'year', 'genre']]
    song_data = json.loads(open('data/song_data.json').read())

    data = []
    for i in range(len(orig_data)):
        entry = orig_data.loc[i, :]
        track = entry['track_id']
        new_entry = [entry['duration'], entry['year'], song_data[track]['parts_of_speech']['noun'],
                     song_data[track]['parts_of_speech']['verb'],
                     song_data[track]['parts_of_speech']['adj'],
                     song_data[track]['parts_of_speech']['adv']]
        data.append(new_entry)

    cov = np.cov(data, rowvar=False)
    print(cov)

    coeff = np.corrcoef(data, rowvar=False)
    print()
    print(coeff)


if __name__ == "__main__":
    main()
