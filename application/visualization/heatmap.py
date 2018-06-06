#! /usr/bin/env python3
""" 
script for generating heatmap of correlation

"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




def main():
    """The executable to read in the specified data file and perform the
    investigatory operations."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        type=argparse.FileType("r"),
        help="The path of the data file to read in.")

    args = parser.parse_args()

    #read the dataframe in memory friendly chunks
    data_frame = pd.read_pickle(args.input_file.name).infer_objects()
    genre_list = ["Punk", "Electronic","RnB", "Rap", "Country", "Metal", "Pop", "Rock"]
    data_frame = data_frame[data_frame["genre"].isin(genre_list)]
    print(data_frame.columns)

    labels = data_frame["genre"]
    data_frame = data_frame.drop(columns=["genre", "release_year"])

    #https://seaborn.pydata.org/examples/many_pairwise_correlations.html
    sns.set(style="white")
    # Compute the correlation matrix
    corr = data_frame.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                        square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()

if __name__ == "__main__":
    main()
