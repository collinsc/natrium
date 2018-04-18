import nltk 
import util.top_words as top_words


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


