import pandas as pd
import numpy as np
import re
from nltk import word_tokenize, sent_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

class TextFeatures(object):
    def __init__:
        pass

'''
-------------------- Sentence Features --------------------
'''

    def sentence_position(self):
        '''
        How will I weight these scores? 
        '''
        pass

    def num_words(self):
        pass

    def term_frequencies(self, sentence, topic, key_words):
        word_appear = 0
        for word in sentence:
            if word in key_words[topic]:
                word_appear += 1
        return word_appear

    def named_entities(self):
        pass

    def presence_of_verb(self):
        pass

    def sentence_length(self):
        pass


'''
-------------------- Word Features --------------------
'''

    def term_frequency(self):
        pass

    def word_length(self):
        pass

    def parts_of_speech(self):
        pass

    def word_familiarity(self):
        pass

    def heading_occurence(self):
        pass
