import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from nltk import word_tokenize, sent_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

class Summarizer(object):
    def __init__(self, bagofwords, sentences,vectorizer, key_words, topic, title = None):
        self.bagofwords = bagofwords
        self.sentences = sentences
        self.vectorizer = vectorizer
        self.key_words = key_words
        self.summary = ''
        self.title = title
        self.sentences = sentences
        self.score = 0.0
        self.reduction = 0.0
        self.sentence_scores = []
        self.sentence_plus_score = []
        self.topic = topic
        self.summary_array = np.array([])
        self.clean_summary_array = np.array([])

    def get_sentence_scores(self):
        '''
        Calculating a sentence score based off keywords from topic modeling
        using a simple # of appearances over length of sentence score.
        '''

        for sentence in self.sentences:
            word_appear = 0
            for word in sentence:
                if word in self.key_words[self.topic]:
                    word_appear += 1
            self.score = word_appear/float(len(sentence))
            self.sentence_scores.append(self.score)

    def summarize(self):
        '''
        Take the sentence scores. Pick highest scoring sentence to form summary.
        '''
        # how do I pick the top sentences and then maintain the order?
        self.sentence_scores = np.array(self.sentence_scores)
        self.sentences = np.array(self.sentences)
        sort_idx = np.argsort(self.sentence_scores)[::-1]
        # cumulative_importance = np.cumsum(self.sentence_scores[sort_idx]/float(np.sum(self.sentence_scores)))
        # # print cumulative_importance
        top_n = np.where(self.sentence_scores > 0.05)
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
        return clean_summary_array
