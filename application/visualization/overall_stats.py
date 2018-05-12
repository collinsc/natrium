#!/usr/bin/env python3
"""Tool for visulizing the genres of a data series
see -h for help"""

import argparse
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import json

#  
def main():
    """Generates graphs for statistics on the genres"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        type=argparse.FileType("r"),
        help="The path of the file to read in.")
    args = parser.parse_args()

    with open(args.input_file.name) as data_file:    
        data = json.load(data_file)
    
    count_data = []
    rhyme_data = []
    noun_data = []
    verb_data = []
    adjective_data = []
    adverb_data = []
    year_data = []
    duration_data = []
    large_genres = ['Punk', 'Electronic', 'RnB', 'Rap', 'Country', 'Metal', 'Pop', 'Rock']
    
    # construct individual genre data
    rhyme_data_genres = dict(zip(large_genres, [[] for x in large_genres]))
    
    # construct individual genre data
    noun_data_genres = dict(zip(large_genres, [[] for x in large_genres]))
    
    # construct individual genre data
    verb_data_genres = dict(zip(large_genres, [[] for x in large_genres]))
    
    # construct individual genre data
    adjective_data_genres = dict(zip(large_genres, [[] for x in large_genres]))
    
    # construct individual genre data
    adverb_data_genres = dict(zip(large_genres, [[] for x in large_genres]))
    
    # construct individual genre data
    year_data_genres = dict(zip(large_genres, [[] for x in large_genres]))
    
    # construct individual genre data
    duration_data_genres = dict(zip(large_genres, [[] for x in large_genres]))
    
    
    for song_name in data:
        song = data[song_name]
        if(song['genre'] not in large_genres):
            continue
        
        count_data.append(song['word_count'])
        
        # skip songs with not enough words
        if(song['word_count'] < 50):
            continue
        
        rhyme = song['rhyme_value']
        rhyme_data.append(rhyme)
        rhyme_data_genres[song['genre']].append(rhyme)
        
        noun = song['parts_of_speech']['noun']
        noun_data.append(noun)
        noun_data_genres[song['genre']].append(noun)
        
        verb = song['parts_of_speech']['verb']
        verb_data.append(verb)
        verb_data_genres[song['genre']].append(verb)
        
        adjective = song['parts_of_speech']['adj']
        adjective_data.append(adjective)
        adjective_data_genres[song['genre']].append(adjective)
        
        adverb = song['parts_of_speech']['adv']
        adverb_data.append(adverb)
        adverb_data_genres[song['genre']].append(adverb)
        
        year = song['release_year']
        if(year > 1750): # get rid of some big outliers (or is it missing data?)
            year_data.append(year)
            year_data_genres[song['genre']].append(year)
        
        duration = song['duration']
        duration_data.append(duration)
        duration_data_genres[song['genre']].append(duration)
        
    
    plt.figure(figsize=(6.4,4), dpi=100)
    plt.hist(count_data, 10, (0, 1000))
    plt.title('Word Counts')
    plt.ylabel('song count')
    plt.xlabel('word count')
    plt.savefig('graphs/counts_overall.png')
    
     
    plt.figure(figsize=(6.4,4), dpi=100)
    plt.hist(rhyme_data, 10, (0, 3))
    plt.title('Rhyming Score')
    plt.ylabel('song count')
    plt.xlabel('rhyming score')
    plt.savefig('graphs/rhymes_overall.png')
    
     
    plt.figure(figsize=(6.4,4), dpi=100)
    plt.hist(noun_data, 10, (0, .5))
    plt.title('Noun Proportion')
    plt.ylabel('song count')
    plt.xlabel('noun proportion')
    plt.savefig('graphs/nouns_overall.png')
    
    
    plt.figure(figsize=(6.4,4), dpi=100)
    plt.hist(verb_data, 10, (0, .5))
    plt.title('Verb Proportion')
    plt.ylabel('song count')
    plt.xlabel('verb proportion')
    plt.savefig('graphs/verbs_overall.png')
    
    
    plt.figure(figsize=(6.4,4), dpi=100)
    plt.hist(adjective_data, 10, (0, .2))
    plt.title('Adjective Proportion')
    plt.ylabel('song count')
    plt.xlabel('adjective proportion')
    plt.savefig('graphs/adjectives_overall.png')
    
    
    plt.figure(figsize=(6.4,4), dpi=100)
    plt.hist(adverb_data, 10, (0, .3))
    plt.title('Adverb Proportion')
    plt.ylabel('song count')
    plt.xlabel('adverbs proportion')
    plt.savefig('graphs/adverbs_overall.png')
    
    
    plt.figure(figsize=(6.4,4), dpi=100)
    plt.hist(year_data, 10, (1950, 2010))
    plt.title('Release Years')
    plt.ylabel('song count')
    plt.xlabel('release year')
    plt.savefig('graphs/years_overall.png')
    
    
    plt.figure(figsize=(6.4,4), dpi=100)
    plt.hist(duration_data, 10, (0, 800))
    plt.title('Song Duration')
    plt.ylabel('song count')
    plt.xlabel('duration (seconds)')
    plt.savefig('graphs/duration_overall.png')
    
    
    
    
    plt.figure(figsize=(6.4,4), dpi=100)
    plt.boxplot(\
        [rhyme_data_genres[genre] for genre in rhyme_data_genres],\
        labels = [genre for genre in rhyme_data_genres])
    plt.title('Rhyming Score By Genre')
    plt.ylabel('rhyming score')
    plt.savefig('graphs/rhymes_by_genre.png')
    
    
    plt.figure(figsize=(6.4,4), dpi=100)
    plt.boxplot(\
        [noun_data_genres[genre] for genre in noun_data_genres],\
        labels = [genre for genre in noun_data_genres])
    plt.title('Noun Proportion By Genre')
    plt.ylabel('noun proportion')
    plt.savefig('graphs/nouns_by_genre.png')
    
    
    plt.figure(figsize=(6.4,4), dpi=100)
    plt.boxplot(\
        [verb_data_genres[genre] for genre in verb_data_genres],\
        labels = [genre for genre in verb_data_genres])
    plt.title('Verb Proportion By Genre')
    plt.ylabel('verb proportion')
    plt.savefig('graphs/verbs_by_genre.png')
    
    
    plt.figure(figsize=(6.4,4), dpi=100)
    plt.boxplot(\
        [adjective_data_genres[genre] for genre in adjective_data_genres],\
        labels = [genre for genre in adjective_data_genres])
    plt.title('Adjective Proportion By Genre')
    plt.ylabel('adjective proportion')
    plt.savefig('graphs/adjectives_by_genre.png')
    
    
    plt.figure(figsize=(6.4,4), dpi=100)
    plt.boxplot(\
        [adverb_data_genres[genre] for genre in adverb_data_genres],\
        labels = [genre for genre in adverb_data_genres])
    plt.title('Adverb Proportion By Genre')
    plt.ylabel('adverb proportion')
    plt.savefig('graphs/adverbs_by_genre.png')
    
    
    plt.figure(figsize=(6.4,4), dpi=100)
    plt.boxplot(\
        [year_data_genres[genre] for genre in year_data_genres],\
        labels = [genre for genre in year_data_genres])
    plt.title('Release Years By Genre')
    plt.ylabel('release year')
    plt.savefig('graphs/years_by_genre.png')
    
    
    plt.figure(figsize=(6.4,4), dpi=100)
    plt.boxplot(\
        [duration_data_genres[genre] for genre in duration_data_genres],\
        labels = [genre for genre in duration_data_genres])
    plt.title('Durations By Genre')
    plt.ylabel('duration (seconds)')
    plt.savefig('graphs/durations_by_genre.png')


    
    
    




if __name__ == "__main__":
    main()
