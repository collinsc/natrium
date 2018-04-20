import pandas as pd
import sys
import numpy
import json
import util.top_words as top_words
import nltk
import util.nltk_helper as nltk_helper
from collections import Counter

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
    return {
        'frequencies': hist[0].tolist(),
        'bin_edges': hist[1].tolist()
    }

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
    genre_parts = dict(zip(tag_list, [[] for _ in range(len(tag_list))]))

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
            genre_parts[tag].append(frequency)

        song_data[song_name]['parts_of_speech'] = dict(zip(tag_list, frequencies))

    for tag in tag_list:
        genre_data[tag.lower() + '_hist'] = __get_frequency_histogram(genre_parts[tag])


# save durations of songs in the songs and a histogram in the genre
def calculate_duration(frame, genre_data, song_data):
    # limit columns to the ones we care about
    frame = frame[['track_id', 'duration']]
    durations = list(frame['duration'])

    hist = numpy.histogram(durations)

    genre_data['duration_hist'] = {
        'frequencies': hist[0].tolist(),
        'bin_edges': hist[1].tolist()
    }

    for index, song in frame.iterrows():
        song_name = song['track_id']
        song_data[song_name]['duration'] = song['duration']


# calculates the top words for a genre and saves them to the genre
def calculate_popular_words(frame, genre_data, song_data):
    word_popularity = Counter()
    song_count = 0
    for index, song in frame.iterrows():
        song_count += 1
        lyrics = __get_lyrics(song)
        for word in lyrics:
            word_popularity[word[0]] += word[1]

    # extract top words and convert to percentage of total words
    top_lyrics = word_popularity.most_common(25)
    genre_data['top_words'] = [{'word':x[0], 'avg_freq':x[1]/song_count/__get_word_count(lyrics)} for x in top_lyrics]


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
        song_data[song_name]['rhyme_value'] = rhyme_value

    hist = numpy.histogram(rhyme_frequencies)
    genre_data['rhyme__hist'] = {
        'frequencies': hist[0].tolist(),
        'bin_edges': hist[1].tolist()
    }


def main():
    genre_data = {}
    song_data = {}

    print('reading csv file from stdin')
    data_frame = pd.read_csv(sys.stdin)

    # sort by genre
    print('sorting by genre')
    for genre, frame in data_frame.groupby('genre'):
         print('processing genre called ' + genre)
         # initialize genre structs
         genre_data[genre] = {}

         # intialize song structs
         for index, song in frame.iterrows():
             song_name = song['track_id']
             song_data[song_name] = {'genre': genre}
         
         # calculate all the interesting features
         calculate_duration(frame, genre_data[genre], song_data)
         print('25%')
         calculate_word_tags(frame, genre_data[genre], song_data)
         print('50%')
         calculate_popular_words(frame, genre_data[genre], song_data)
         print('75%')
         calculate_rhymes(frame, genre_data[genre], song_data)
         print('100%')
         
    with open('genre_data.txt', 'w') as f:
        f.write(json.dumps(genre_data, indent=4))

    with open('song_data.txt', 'w') as f:
        f.write(json.dumps(song_data, indent=4))


if __name__ == '__main__':
    main()

