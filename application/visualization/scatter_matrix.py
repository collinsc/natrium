#!/usr/bin/env python3
"""Tool for generating a scatter matrix for iteration 2 of the project."""

import pandas.core.frame
import matplotlib as mpl

mpl.use('Agg')  # Necessary so we can work without Tkinter
import matplotlib.pyplot as plt
import numpy as np
import json


def main():
    ticklabel_fontsize = 12
    label_fontsize = 24
    data_labels = ["Duration (seconds)", "Year", "Noun Proportion", "Verb Proportion",
                   "Adjective Proportion", "Adverb Proportion", "Rhyme Score"]
    genres = ["Punk", "Electronic", "RnB", "Rap", "Country", "Metal", "Pop", "Rock"]
    genre_colors = ['#000000', '#FFFF00', '#00FF00', '#008000', '#00FFFF', '#0000FF', '#FF00FF',
                    '#FF0000']
    data = {}  # Dict: Genre -> List of data lists in order of labels

    orig_data = pandas.read_pickle('data/all.pkl')[['track_id', 'duration', 'year', 'genre']]
    song_data = json.loads(open('data/song_data.json').read())

    for genre in genres:
        data[genre] = []
        subset = orig_data[orig_data['genre'] == genre]

        # Add duration and year
        data[genre].append(list(subset['duration']))
        data[genre].append(list(subset['year']))

        # Add the proportions
        noun_props = []
        verb_props = []
        adj_props = []
        adv_props = []
        rhyming_props = []
        for track in subset['track_id']:
            noun_props.append(song_data[track]['parts_of_speech']['noun'])
            verb_props.append(song_data[track]['parts_of_speech']['verb'])
            adj_props.append(song_data[track]['parts_of_speech']['adj'])
            adv_props.append(song_data[track]['parts_of_speech']['adv'])
            rhyming_props.append(song_data[track]['rhyme_value'])

        data[genre].append(noun_props)
        data[genre].append(verb_props)
        data[genre].append(adj_props)
        data[genre].append(adv_props)
        data[genre].append(rhyming_props)

    # Cut down data
    for genre in genres:
        indices = np.random.permutation(len(data[genre][0]))[0:250]

        zipped = list(zip(data[genre][0], data[genre][1], data[genre][2], data[genre][3],
                          data[genre][4], data[genre][5], data[genre][6]))
        filtered = []
        for index in indices:
            filtered.append(zipped[index])

        data[genre][0], data[genre][1], data[genre][2], data[genre][3], data[genre][4] \
            , data[genre][5], data[genre][6] = zip(*filtered)

    plt.close()  # Just in case

    fig = plt.figure(figsize=(16, 12))

    # Plot the data labels
    for i in range(len(data_labels)):
        sub = plt.subplot(len(data_labels), len(data_labels), i * len(data_labels) + i + 1)
        sub.text(0.5, 0.5, data_labels[i].replace(' ', '\n'), horizontalalignment='center',
                 verticalalignment='center', fontsize=label_fontsize)
        sub.set_frame_on(False)
        sub.get_xaxis().set_visible(False)
        sub.get_yaxis().set_visible(False)

    # Plot the data itself
    for x in range(len(data_labels)):
        for y in range(x + 1, len(data_labels)):
            sub = plt.subplot(len(data_labels), len(data_labels), y * len(data_labels) + x + 1)
            sub.set_frame_on(True)
            for i in range(len(genres)):
                genre = genres[i]

                x_data = data[genre][x]
                y_data = data[genre][y]

                # Trim year data down
                if data_labels[x] == 'Year':
                    merged = zip(data[genre][x], data[genre][y])
                    filtered = filter(lambda pair: pair[0] > 0, merged)
                    x_data, y_data = zip(*filtered)

                if data_labels[y] == 'Year':
                    merged = zip(data[genre][x], data[genre][y])
                    filtered = filter(lambda pair: pair[1] > 0, merged)
                    x_data, y_data = zip(*filtered)

                sub.plot(x_data, y_data, markeredgecolor=genre_colors[i],
                         marker='o', linestyle='None', markersize=0.5)

                for tick in sub.get_xaxis().get_major_ticks():
                    tick.label.set_fontsize(ticklabel_fontsize)
                for tick in sub.get_yaxis().get_major_ticks():
                    tick.label.set_fontsize(ticklabel_fontsize)

    # Make the legend
    plt.figlegend(labels=genres, fontsize=24, markerscale=24)

    plt.suptitle("Scatter matrix", fontsize=48)

    fig.tight_layout()
    plt.savefig("visualizations/scatter_matrix.png", dpi=480)


if __name__ == "__main__":
    main()
