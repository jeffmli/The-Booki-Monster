import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk import word_tokenize, sent_tokenize
import random

class RandomSummarizer(object):
    def __init__(self, sentences, summary_length):
        self.sentences = sentences
        self.summary_length = summary_length
        self.random_summary = []

    def sentence_generator(self):
        '''
        Randomly chooses sentences from sentence variable.
        '''
        # random.choice(chapter1_sentences_cleaned)
        for i in range(self.summary_length):
            chosen_sentence = random.choice(self.sentences)
            # if chosen_sentence not in self.random_summary:
            self.random_summary.append(chosen_sentence)

    def format_summary(self):
        '''
        OUTPUT: Random Summary formatted into string.
        '''
        self.random_summary = [''.join(sentence) for sentence in self.random_summary]
        self.random_summary = ' '.join(self.random_summary)
        return self.random_summary
