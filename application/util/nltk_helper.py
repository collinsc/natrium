import pronouncing
import nltk 
import util.top_words as top_words


print('loading nltk.corpus.brown')

# this takes a long time so eventually replace it with a pregenerated file that covers all 5,000 of our words
__wordtags = nltk.ConditionalFreqDist((w.lower(), t)
    for w, t in nltk.corpus.brown.tagged_words(tagset="universal"))



# get the most likely word tag for this situation
# it does this by simply taking the part of speech
# the word is used most as. Returns None if it failed
def get_word_tag(word):
    try:
        return __wordtags[word.lower()].max()
    except ValueError:
        return None

print('precomputing rhymes')


# for whatever reason pronouncing.rhymes() takes a very long time
# extracting what we need before hand provides a 30x speed up
__rhymes = [pronouncing.rhymes(x) for x in top_words.unstemmed]
__rhyme_flattend = []
for word, rhymes in zip(top_words.unstemmed, __rhymes):
    for rhyme in rhymes:
        __rhyme_flattend.append( (word, rhyme) )

__rhyme_set = frozenset(__rhyme_flattend)

def check_rhyme(word1, word2):
    return (word1, word2) in __rhyme_set


