#!/usr/bin/env python3
"""Tool for visulizing the genres of a data series
see -h for help"""

import argparse
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import json

def main():
    """Generates graphs for statistics on the genres"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        type=argparse.FileType("r"),
        help="The path of the file to read in.")
    args = parser.parse_args()

    data = pd.read_pickle(args.input_file.name)
    
    count_data = []
    rhyme_data = []
    noun_data = []
    verb_data = []
    adjective_data = []
    adverb_data = []
    year_data = []
    duration_data = []
    punk_data = []
    electronic_data = []
    rnb_data = []
    rap_data = []
    country_data = []
    metal_data = []
    pop_data = []
    rock_data = []
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
    
    # construct individual genre data
    punk_data_genres = dict(zip(large_genres, [[] for x in large_genres]))
    
    # construct individual genre data
    electronic_data_genres = dict(zip(large_genres, [[] for x in large_genres]))
    
    # construct individual genre data
    rnb_data_genres = dict(zip(large_genres, [[] for x in large_genres]))
    
    # construct individual genre data
    rap_data_genres = dict(zip(large_genres, [[] for x in large_genres]))
    
    # construct individual genre data
    country_data_genres = dict(zip(large_genres, [[] for x in large_genres]))
    
    # construct individual genre data
    metal_data_genres = dict(zip(large_genres, [[] for x in large_genres]))
    
    # construct individual genre data
    pop_data_genres = dict(zip(large_genres, [[] for x in large_genres]))
    
    # construct individual genre data
    rock_data_genres = dict(zip(large_genres, [[] for x in large_genres]))
    
    
    for index, song in data.iterrows():
        count_data.append(song['word_count'])
        
        # skip songs with not enough words
        #if(song['word_count'] < 50):
        #    continue
        
        rhyme = song['rhyme_value']
        rhyme_data.append(rhyme)
        rhyme_data_genres[song['genre']].append(rhyme)
        
        noun = song['noun']
        noun_data.append(noun)
        noun_data_genres[song['genre']].append(noun)
        
        verb = song['verb']
        verb_data.append(verb)
        verb_data_genres[song['genre']].append(verb)
        
        adjective = song['adj']
        adjective_data.append(adjective)
        adjective_data_genres[song['genre']].append(adjective)
        
        adverb = song['adv']
        adverb_data.append(adverb)
        adverb_data_genres[song['genre']].append(adverb)
        
        year = song['release_year']
        if(year > 1750): # get rid of some big outliers (or is it missing data?)
            year_data.append(year)
            year_data_genres[song['genre']].append(year)
        
        duration = song['duration']
        duration_data.append(duration)
        duration_data_genres[song['genre']].append(duration)
        

        punk = song['word_pop_punk']
        punk_data.append(punk)
        punk_data_genres[song['genre']].append(punk)

        electronic = song['word_pop_electronic']
        electronic_data.append(electronic)
        electronic_data_genres[song['genre']].append(electronic)

        rnb = song['word_pop_rnb']
        rnb_data.append(rnb)
        rnb_data_genres[song['genre']].append(rnb)

        rap = song['word_pop_rap']
        rap_data.append(rap)
        rap_data_genres[song['genre']].append(rap)

        country = song['word_pop_country']
        country_data.append(country)
        country_data_genres[song['genre']].append(country)

        metal = song['word_pop_metal']
        metal_data.append(metal)
        metal_data_genres[song['genre']].append(metal)

        pop = song['word_pop_pop']
        pop_data.append(pop)
        pop_data_genres[song['genre']].append(pop)

        rock = song['word_pop_rock']
        rock_data.append(rock)
        rock_data_genres[song['genre']].append(rock)
        
    
    plt.figure(figsize=(6.4,3.8), dpi=100)
    plt.hist(count_data, 10, (0, 1000))
    plt.title('Word Counts')
    plt.ylabel('song count')
    plt.xlabel('word count')
    plt.savefig('../graphs/counts_overall.png')
    plt.close()
    
     
    plt.figure(figsize=(6.4,3.8), dpi=100)
    plt.hist(rhyme_data, 10, (0, 1))
    plt.title('Rhyming Score')
    plt.ylabel('song count')
    plt.xlabel('rhyming score')
    plt.savefig('../graphs/rhymes_overall.png')
    plt.close()
    
     
    plt.figure(figsize=(6.4,3.8), dpi=100)
    plt.hist(noun_data, 10, (0, .5))
    plt.title('Noun Proportion')
    plt.ylabel('song count')
    plt.xlabel('noun proportion')
    plt.savefig('../graphs/nouns_overall.png')
    plt.close()
    
    
    plt.figure(figsize=(6.4,3.8), dpi=100)
    plt.hist(verb_data, 10, (0, .5))
    plt.title('Verb Proportion')
    plt.ylabel('song count')
    plt.xlabel('verb proportion')
    plt.savefig('../graphs/verbs_overall.png')
    plt.close()
    
    
    plt.figure(figsize=(6.4,3.8), dpi=100)
    plt.hist(adjective_data, 10, (0, .2))
    plt.title('Adjective Proportion')
    plt.ylabel('song count')
    plt.xlabel('adjective proportion')
    plt.savefig('../graphs/adjectives_overall.png')
    plt.close()
    
    
    plt.figure(figsize=(6.4,3.8), dpi=100)
    plt.hist(adverb_data, 10, (0, .3))
    plt.title('Adverb Proportion')
    plt.ylabel('song count')
    plt.xlabel('adverbs proportion')
    plt.savefig('../graphs/adverbs_overall.png')
    plt.close()
    
    
    plt.figure(figsize=(6.4,3.8), dpi=100)
    plt.hist(year_data, 10, (1950, 2010))
    plt.title('Release Years')
    plt.ylabel('song count')
    plt.xlabel('release year')
    plt.savefig('../graphs/years_overall.png')
    plt.close()
    
    
    plt.figure(figsize=(6.4,3.8), dpi=100)
    plt.hist(duration_data, 10, (0, 800))
    plt.title('Song Duration')
    plt.ylabel('song count')
    plt.xlabel('duration (seconds)')
    plt.savefig('../graphs/duration_overall.png')
    plt.close()
    
    

    plt.figure(figsize=(6.4,3.8), dpi=100)
    plt.hist(punk_data, 10, (0, .2))
    plt.title('Punk Dictionary Count')
    plt.savefig('../graphs/punk_dict_overall.png')
    plt.close()

    plt.figure(figsize=(6.4,3.8), dpi=100)
    plt.hist(electronic_data, 10, (0, .06))
    plt.title('Electronic Dictionary Count')
    plt.savefig('../graphs/electronic_dict_overall.png')
    plt.close()

    plt.figure(figsize=(6.4,3.8), dpi=100)
    plt.hist(rnb_data, 10, (0, .3))
    plt.title('Rnb Dictionary Count')
    plt.savefig('../graphs/rnb_dict_overall.png')
    plt.close()

    plt.figure(figsize=(6.4,3.8), dpi=100)
    plt.hist(rap_data, 10, (0, .3))
    plt.title('Rap Dictionary Count')
    plt.savefig('../graphs/rap_dict_overall.png')
    plt.close()

    plt.figure(figsize=(6.4,3.8), dpi=100)
    plt.hist(country_data, 10, (0, .06))
    plt.title('Country Dictionary Count')
    plt.savefig('../graphs/country_dict_overall.png')
    plt.close()

    plt.figure(figsize=(6.4,3.8), dpi=100)
    plt.hist(metal_data, 10, (0, .2))
    plt.title('Metal Dictionary Count')
    plt.savefig('../graphs/metal_dict_overall.png')
    plt.close()

    plt.figure(figsize=(6.4,3.8), dpi=100)
    plt.hist(pop_data, 10, (0, .06))
    plt.title('Pop Dictionary Count')
    plt.savefig('../graphs/pop_dict_overall.png')
    plt.close()

    plt.figure(figsize=(6.4,3.8), dpi=100)
    plt.hist(rock_data, 10, (0, .3))
    plt.title('Rock Dictionary Count')
    plt.savefig('../graphs/rock_dict_overall.png')
    plt.close()
    
    
    
    
    plt.figure(figsize=(6.4,3.8), dpi=100)
    plt.boxplot(\
        [rhyme_data_genres[genre] for genre in rhyme_data_genres],\
        labels = [genre for genre in rhyme_data_genres])
    plt.title('Rhyming Score By Genre')
    plt.ylabel('rhyming score')
    plt.savefig('../graphs/rhymes_by_genre.png')
    plt.close()
    
    
    plt.figure(figsize=(6.4,3.8), dpi=100)
    plt.boxplot(\
        [noun_data_genres[genre] for genre in noun_data_genres],\
        labels = [genre for genre in noun_data_genres])
    plt.title('Noun Proportion By Genre')
    plt.ylabel('noun proportion')
    plt.savefig('../graphs/nouns_by_genre.png')
    plt.close()
    
    
    plt.figure(figsize=(6.4,3.8), dpi=100)
    plt.boxplot(\
        [verb_data_genres[genre] for genre in verb_data_genres],\
        labels = [genre for genre in verb_data_genres])
    plt.title('Verb Proportion By Genre')
    plt.ylabel('verb proportion')
    plt.savefig('../graphs/verbs_by_genre.png')
    plt.close()
    
    
    plt.figure(figsize=(6.4,3.8), dpi=100)
    plt.boxplot(\
        [adjective_data_genres[genre] for genre in adjective_data_genres],\
        labels = [genre for genre in adjective_data_genres])
    plt.title('Adjective Proportion By Genre')
    plt.ylabel('adjective proportion')
    plt.savefig('../graphs/adjectives_by_genre.png')
    plt.close()
    
    
    plt.figure(figsize=(6.4,3.8), dpi=100)
    plt.boxplot(\
        [adverb_data_genres[genre] for genre in adverb_data_genres],\
        labels = [genre for genre in adverb_data_genres])
    plt.title('Adverb Proportion By Genre')
    plt.ylabel('adverb proportion')
    plt.close()
    plt.savefig('../graphs/adverbs_by_genre.png')
    plt.close()
    
    
    plt.figure(figsize=(6.4,3.8), dpi=100)
    plt.boxplot(\
        [year_data_genres[genre] for genre in year_data_genres],\
        labels = [genre for genre in year_data_genres])
    plt.title('Release Years By Genre')
    plt.ylabel('release year')
    plt.savefig('../graphs/years_by_genre.png')
    plt.close()
    
    
    plt.figure(figsize=(6.4,3.8), dpi=100)
    plt.boxplot(\
        [duration_data_genres[genre] for genre in duration_data_genres],\
        labels = [genre for genre in duration_data_genres])
    plt.title('Durations By Genre')
    plt.ylabel('duration (seconds)')
    plt.savefig('../graphs/durations_by_genre.png')
    plt.close()



    plt.figure(figsize=(6.4,3.8), dpi=100)
    plt.boxplot(\
        [punk_data_genres[genre] for genre in punk_data_genres],\
        labels = [genre for genre in punk_data_genres])
    plt.title('Punk Dictionary Count By Genre')
    plt.ylabel('dictionary proportion')
    plt.savefig('../graphs/punk_by_genre.png')
    plt.close()

    plt.figure(figsize=(6.4,3.8), dpi=100)
    plt.boxplot(\
        [electronic_data_genres[genre] for genre in electronic_data_genres],\
        labels = [genre for genre in electronic_data_genres])
    plt.title('Electronic Dictionary Count By Genre')
    plt.ylabel('dictionary proportion')
    plt.savefig('../graphs/electronic_by_genre.png')
    plt.close()
    
    plt.figure(figsize=(6.4,3.8), dpi=100)
    plt.boxplot(\
        [rnb_data_genres[genre] for genre in rnb_data_genres],\
        labels = [genre for genre in rnb_data_genres])
    plt.title('RnB Dictionary Count By Genre')
    plt.ylabel('dictionary proportion')
    plt.savefig('../graphs/rnb_by_genre.png')
    plt.close()
    
    plt.figure(figsize=(6.4,3.8), dpi=100)
    plt.boxplot(\
        [rap_data_genres[genre] for genre in rap_data_genres],\
        labels = [genre for genre in rap_data_genres])
    plt.title('Rap Dictionary Count By Genre')
    plt.ylabel('dictionary proportion')
    plt.savefig('../graphs/rap_by_genre.png')
    plt.close()
    
    plt.figure(figsize=(6.4,3.8), dpi=100)
    plt.boxplot(\
        [country_data_genres[genre] for genre in country_data_genres],\
        labels = [genre for genre in country_data_genres])
    plt.title('Country Dictionary Count By Genre')
    plt.ylabel('dictionary proportion')
    plt.savefig('../graphs/country_by_genre.png')
    plt.close()
    
    plt.figure(figsize=(6.4,3.8), dpi=100)
    plt.boxplot(\
        [metal_data_genres[genre] for genre in metal_data_genres],\
        labels = [genre for genre in metal_data_genres])
    plt.title('Metal Dictionary Count By Genre')
    plt.ylabel('dictionary proportion')
    plt.savefig('../graphs/metal_by_genre.png')
    plt.close()
    
    plt.figure(figsize=(6.4,3.8), dpi=100)
    plt.boxplot(\
        [pop_data_genres[genre] for genre in pop_data_genres],\
        labels = [genre for genre in pop_data_genres])
    plt.title('Pop Dictionary Count By Genre')
    plt.ylabel('dictionary proportion')
    plt.savefig('../graphs/pop_by_genre.png')
    plt.close()
    
    plt.figure(figsize=(6.4,3.8), dpi=100)
    plt.boxplot(\
        [rock_data_genres[genre] for genre in rock_data_genres],\
        labels = [genre for genre in rock_data_genres])
    plt.title('Rock Dictionary Count By Genre')
    plt.ylabel('dictionary proportion')
    plt.savefig('../graphs/rock_by_genre.png')
    plt.close()
    
    




if __name__ == "__main__":
    main()
