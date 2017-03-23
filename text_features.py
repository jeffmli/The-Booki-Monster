import pandas as pd
import numpy as np
import re
from nltk import word_tokenize, sent_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from collections import Counter
from spacy.gold import GoldParse
from spacy.language import EntityRecognizer

class TextFeatures(object):
    def __init__(self):

        '''
        -------------------- Sentence Features --------------------
        '''

    def sentence_position_weight(self, documents, sentences):
        '''
        Weights a sentence based on it's position in the chapter.
        '''
        sentence_probability_dict = dict.fromkeys(documents, 0)
        for position, sentence in enumerate(sentence_probability_dict):
            if float(position)/len(documents)<=0.1:
                sentence_probability_dict[sentence] = 0.17
            elif 0.1 < float(position)/len(documents) <= 0.2:
                sentence_probability_dict[sentence] = 0.23
            elif 0.2 < float(position)/len(documents) <= 0.3:
                sentence_probability_dict[sentence] = 0.14
            elif 0.3 < float(position)/len(documents) <= 0.4:
                sentence_probability_dict[sentence] = 0.08
            elif 0.4 < float(position)/len(documents) <= 0.5:
                sentence_probability_dict[sentence] = 0.05
            elif 0.5 < float(position)/len(documents) <= 0.6:
                sentence_probability_dict[sentence] = 0.04
            elif 0.6 < float(position)/len(documents) <= 0.7:
                sentence_probability_dict[sentence] = 0.06
            elif 0.7 < float(position)/len(documents) <= 0.8:
                sentence_probability_dict[sentence] = 0.04
            elif 0.8 < float(position)/len(documents) <= 0.9:
                sentence_probability_dict[sentence] = 0.04
            elif 0.9 < float(position)/len(documents) <= 1.0:
                sentence_probability_dict[sentence] = 0.15
        return sentence_probability_dict

    def num_words(self, documents):
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

    def named_entities(self):
        '''
        Look at the amount of times a named entity occurs over the entire corpus,
        select the entities that have the highest frequency. Add positive rank to
        the sentence if it contains an important entity.
        '''
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
