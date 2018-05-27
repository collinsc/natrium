import argparse
import pandas as pd
import sys, os
import numpy
from  util import top_words, nltk_helper
import nltk
from collections import Counter
from collections import defaultdict

# helper function to parse lyrics string in the csv
def __get_lyrics(song):
    lyrics_str = song['lyrics']

    # Split into the word-count pairs, and split those pairs into an array as well and convert strings to numbers
    # e.g. [[1,2], [3,1]] is two occurences of the #1 word and one occurence of the #3 word
    lyrics = [[int(y) for y in x.split(':')] for x in lyrics_str.split()]

    # finally convert to zero based so we can index an array
    return [(x[0]-1, x[1]) for x in lyrics]

# helper to get count of lyrics
def __get_word_count(lyrics):
    total = 0
    for word in lyrics:
        total += word[1]
    return total 

# helper to create percentage histogram
def __get_frequency_histogram(values):
    hist = numpy.histogram(values, bins=numpy.linspace(0, 1, 11))
    return { 'frequencies': hist[0].tolist(), 'bin_edges': hist[1].tolist() }

# calculates the parts of speech the words in a song are (as percentage of total words)
# for the song, it saves them like the following:
# parts_of_speech = {
#    noun: .15,
#    verb: .1
# }
# for the genre it saves the histogram of song distributions like so:
# noun_hist = {
#     frequencies: {
#         .21,
#         ...
#         .01
#     }
#     bin_edges: {
#         .0
#         ...
#         1.0
#     }
def calculate_word_tags(frame, genre_data, song_data):
    # limit columns to the ones we care about
    frame = frame[['track_id', 'lyrics']]

    # list of wanted tag types (missing punctuation as that should not be in the dataset it seems)
    tag_list = ['adj', 'adp', 'adv', 'conj', 'det', 'noun', 'num', 'prt', 'pron', 'verb', 'x']
    genre_parts = {tag:[] for tag in tag_list}

    for index, song in frame.iterrows():
        song_name = song['track_id']

        lyrics = __get_lyrics(song)

        speech_counter = Counter()
        for pair in lyrics:
            part_of_speech = nltk_helper.get_word_tag(top_words.unstemmed[pair[0]])
            if(part_of_speech):
                speech_counter[part_of_speech.lower()] += pair[1]

        counts = [speech_counter[x] for x in tag_list]
        frequencies = [x/__get_word_count(lyrics) for x in counts]

        for tag, frequency in zip(tag_list, frequencies):
            song_data.loc[song_name][tag] = frequency
            genre_parts[tag].append(frequency)
        song_data.loc[song_name]['word_count'] = __get_word_count(lyrics) 

    for tag in tag_list:
        genre_data[tag.lower() + '_hist'] = __get_frequency_histogram(genre_parts[tag])


# save durations of songs in the songs and a histogram in the genre
def calculate_duration(frame, genre_data, song_data):
    # limit columns to the ones we care about
    frame = frame[['track_id', 'duration', 'year']]
    durations = list(frame['duration'])

    hist = numpy.histogram(durations)

    genre_data['duration_hist'] = {
        'frequencies': hist[0].tolist(),
        'bin_edges': hist[1].tolist()
    }

    for index, song in frame.iterrows():
        song_name = song['track_id']
        song_data.loc[song_name]['duration'] = song['duration']
        song_data.loc[song_name]['release_year'] = song['year'] # TODO move later to own section?


# calculates the top words for a genre and saves them to the genre
def calculate_popular_words(frame, genre_data, song_data, genre_top_words, valid_genres):
    for index, song in frame.iterrows():
        song_name = song['track_id']
        lyrics = __get_lyrics(song)
        count_per_genre = defaultdict(int)
        for word_idx, count in lyrics:
            for genre in valid_genres:
                count_per_genre[genre] += count if(top_words.stemmed[word_idx] in genre_top_words[genre][0]) else 0
        for genre in valid_genres:
            count_per_genre[genre] /= __get_word_count(lyrics) # normalize
            song_data.loc[song_name]['word_pop_' + genre.lower()] = count_per_genre[genre]


# calculates rhymes based off what the pronouncing library says
# currently this takes an optimistic aproach by matching any possible rhyme.
# for example, 'bow' like something that shoots arrows and 'bow' like to an adoring crowd
# given the bag of words format there is no way to identify which one is intended
# here we match any possible way a spelling could rhyme
# saves the number of rhymes as a percentage (relative to word count) for every song
# and of course a distribution per genre
def calculate_rhymes(frame, genre_data, song_data):
    song_count = 0
    rhyme_frequencies = []
    for index, song in frame.iterrows():
        song_name = song['track_id']
        song_count += 1
        rhyme_value = 0
        lyrics = __get_lyrics(song)
        for idx, word in enumerate(lyrics):
            for word_cmp in lyrics[idx+1:]:
                word_str1 = top_words.unstemmed[word[0]]
                word_str2 = top_words.unstemmed[word_cmp[0]]
                total_count = word[1] + word_cmp[1]
                value = total_count if nltk_helper.check_rhyme(word_str1, word_str2) else 0
                rhyme_value += value

        rhyme_value = rhyme_value / __get_word_count(lyrics)
        rhyme_frequencies.append(rhyme_value)
        song_data.loc[song_name]['rhyme_value'] = rhyme_value

    hist = numpy.histogram(rhyme_frequencies)
    genre_data['rhyme__hist'] = {
        'frequencies': hist[0].tolist(),
        'bin_edges': hist[1].tolist()
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_file",
        type=argparse.FileType("r"),
        help="The path of the file to read in.")

    args = parser.parse_args()


    genre_data = {}
    song_data = {}

    valid_genres = ["Punk", "Electronic", "RnB", "Rap", "Country", "Metal", "Pop", "Rock"]


    print('reading pickle file from ' + args.input_file.name)
    data_frame = pd.read_pickle(args.input_file.name)
    data_frame = data_frame[data_frame['genre'].isin(valid_genres)]

    genre_data = {}
    song_data = pd.DataFrame(index=data_frame["track_id"],columns=["genre", "duration", "release_year",
        'adj', 'adp', 'adv', 'conj', 'det', 'noun', 'num', 'prt', 'pron', 
        'verb', 'x', "word_count", "rhyme_value", 'word_pop_punk',
        'word_pop_electronic', 'word_pop_rnb', 'word_pop_rap', 'word_pop_country',
        'word_pop_pop', 'word_pop_rock']) 


    # sort by genre
    print('sorting by genre')
    groups = data_frame.groupby('genre')
    
    # first pass collect word counts per genre. Needed for getting unique and popular
    # words between genres. (e.g. truck is rather unqiue and popular to country)
    # stored as such:
    # {
    #   word1 : {genre1: .4, genre2: .5, ...},
    #   word2 : {genre1: .7, genre2: .1, ...},
    #   ...
    #   wordn : {genre1: .4, genre2: .5, ...}
    # }
    # where the numbers after the genre keys are the average percentage of that
    # word in that genre.
    genre_dict_list = [defaultdict(int) for x in range(len(top_words.stemmed))]
    word_genre_counts = dict(zip(top_words.stemmed, genre_dict_list))
    for genre, frame in groups:
        print('calculating genre uses per word for ' + genre)
        # limit columns to the ones we care about
        frame = frame[['track_id', 'lyrics']]

        song_count = 0
        for index, song in frame.iterrows():
            lyrics = __get_lyrics(song)
            for word_idx, count in lyrics:
                word = top_words.stemmed[word_idx]
                
                # add the number of occurences of this word, normalized
                # to the total number of words in the song (to the range [0, 1])
                word_genre_counts[word][genre] += count/__get_word_count(lyrics)
            
            song_count += 1
        
        # get the average of normalized word counts 
        for word in top_words.stemmed:
            word_genre_counts[word][genre] /= song_count

    genre_top_words = defaultdict(list)
    for word_idx in word_genre_counts:
        word = word_genre_counts[word_idx] # remember: of the form {Rock: .1, Country: .2, ...}
        
        # list of keys sorted by value
        ordered_keys = sorted(word, key=word.__getitem__)
        
        first_genre = ordered_keys[-1] # genre that uses the word the most
        second_genre = ordered_keys[-2] # the genre that uses it the second most
        
        # skip words that are unlikely
        if(word[first_genre] < .001):
            continue
        
        # this disparity ratio will be used to find words that stand out for a genre
        ratio = word[first_genre] / word[second_genre]
        
        # give the useful stuff: which word it is and the ratio above the second highest.
        # Also the follow genre for funzies
        genre_top_words[first_genre].append((word_idx, ratio, second_genre))
    

    # eliminate it to top N words and sort
    for genre in valid_genres:
        N = 5
        genre_top_words[genre].sort(key=lambda word: word[1], reverse=True)
        genre_top_words[genre] = genre_top_words[genre][0:N] # limit it
        genre_top_words[genre] = [x[0] for x in genre_top_words[genre]] # simplify it to just the word

    import json # temp
    print(json.dumps(genre_top_words, indent=4))

    # process features on a per genre basis
    for genre, frame in groups:
         print('processing genre called ' + genre)
         # initialize genre structs
         genre_data[genre] = {}

         # intialize song structs
         for index, song in frame.iterrows():
             song_name = song['track_id']
             song_data.loc[song_name]["genre"] = genre
         
         # calculate all the interesting features
         calculate_popular_words(frame, genre_data[genre], song_data, genre_top_words, valid_genres)
         print('25%')
         calculate_word_tags(frame, genre_data[genre], song_data)
         print('50%')
         calculate_duration(frame, genre_data[genre], song_data)
         print('75%')
         calculate_rhymes(frame, genre_data[genre], song_data)
         print('100%')

    pd.DataFrame.from_dict(genre_data).to_pickle("../genre_data.pkl")
    song_data.infer_objects()
    song_data.to_pickle("../song_data.pkl")
         

if __name__ == '__main__':
    main()

