import pandas as pd
import numpy as np
import re
from nltk import word_tokenize, sent_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from collections import Counter
from nltk.tag import brill
from nltk import pos_tag

class TextFeatures(object):
    def __init__(self):

        '''
        -------------------- Sentence Features --------------------
        '''

    def sentence_position_weight(self, pos_percent):
        '''
        Weights a sentence based on it's position in the chapter.
        '''
        if pos_percent <=0.1:
            return 0.17
        elif 0.1 < pos_percent <= 0.2:
            return 0.23
        elif 0.2 < pos_percent <= 0.3:
            return 0.14
        elif 0.3 < pos_percent <= 0.4:
            return 0.08
        elif 0.4 < pos_percent <= 0.5:
            return 0.05
        elif 0.5 < pos_percent <= 0.6:
            return 0.04
        elif 0.6 < pos_percent <= 0.7:
            return 0.06
        elif 0.7 < pos_percent <= 0.8:
            return 0.04
        elif 0.8 < pos_percent <= 0.9:
            return 0.04
        elif 0.9 < pos_percent <= 1.0:
            return 0.15

    def num_tokens(self, documents):
        '''
        Number of words in sentence
        '''
        num_words_dict = dict.fromkeys(documents, 0)
        for sentence in num_words_dict:
            num_words_dict[sentence] = len(sentence)
        return num_words_dict

    def term_frequencies(self, sentence, topic, key_words):
        word_appear = 0
        for word in sentence:
            if word in key_words[topic]:
                word_appear += 1
        return word_appear

    def presence_of_verb(self, sentence):
        '''
        If sentence contains a verb, upweight the score of the sentence.
        '''
        verb_type = ['VB','VBZ','VBN','VBG','VBD']
        # text = word_tokenize(sentence)
        tag = pos_tag(sentence)
        verb_count = 0
        for word in tag:
            if any(verb in word for verb in verb_type):
                verb_count += 1
        return verb_count

    '''
    -------------------- Word Features --------------------
    '''

    def parts_of_speech(self):
        pass

    def word_familiarity(self):
        pass
