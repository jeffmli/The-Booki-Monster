import pandas as pd
import numpy as np
import re
from nltk import word_tokenize, sent_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from text_features import TextFeatures

class Summarizer(object):
    def __init__(self, documents, sentences,vectorizer, key_words, topic):
        self.documents = documents
        self.sentences = sentences
        self.vectorizer = vectorizer
        self.key_words = key_words
        self.score = 0.0
        self.sentence_scores = []
        self.topic = topic
        self.summary_array = np.array([])
        self.sentence_idx = np.array([])

    '''
    -------------------------------Feature Engineering-------------------------------
    '''

    def get_sentence_scores(self):
        '''
        Calculate sentence scores from engineered features
        '''
        text_features = TextFeatures()
        for sentence in self.sentences:
            '''
            Position-Weight Feature
            '''
            position_dict = text_features.sentence_position_weight(self.documents, self.sentences)
            '''
            Num Words Feature
            '''
            num_words = len(sentence)
            '''
            Term-Frequency Feature
            '''
            self.score = text_features.term_frequencies(sentence, self.topic, self.key_words)
            self.sentence_scores.append(self.score)

    '''
    -------------------------------Summarizing-------------------------------
    '''
    # def sentence_rank(self):
    #     '''
    #     Ranking the sentences in order of importance
    #     '''
    #     self.sentence_scores = np.array(self.sentence_scores)
    #     self.sentences = np.array(self.sentences)
    #
    #     sort_idx = np.argsort(self.sentence_scores)[::-1]
    #     cumulative_importance = np.cumsum(self.sentence_scores[sort_idx]/float(np.sum(self.sentence_scores)))
    #
    #     top_n = np.where(cumulative_importance > .9)
    #     important_sentence_idx = sort_idx[top_n]
    #     sentence_idx = np.sort(important_sentence_idx)


    def summarize(self):
        '''
        Create sentence rank. Pick highest scoring sentence(s) to form summary.
        Note: Divide out sentence rank into another function if I have more time.
        '''
        # how do I pick the top sentences and then maintain the order?
        self.sentence_scores = np.array(self.sentence_scores)
        self.sentences = np.array(self.sentences)

        sort_idx = np.argsort(self.sentence_scores)[::-1]
        cumulative_importance = np.cumsum(self.sentence_scores[sort_idx]/float(np.sum(self.sentence_scores)))

        top_n = np.where(cumulative_importance > 0.9)
        important_sentence_idx = sort_idx[top_n]
        sentence_idx = np.sort(important_sentence_idx)

        summary_array = self.sentences[sentence_idx]
        self.summary_array = [' '.join(sentence) for sentence in summary_array]


    def format_summary(self):
        '''
        Formatting the summary into a readable format.
        '''
        clean_summary_array = []
        for sentence in self.summary_array:
            sentence = ''.join(sentence[0].upper() + sentence[1:])
            sentence = sentence + "."
            clean_summary_array.append(sentence)
        clean_summary_array = ' '.join(clean_summary_array).replace(',', '.')
        return clean_summary_array, self.summary_array
