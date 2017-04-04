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
        # random.choice(chapter1_sentences_cleaned)
        for i in range(self.summary_length):
            chosen_sentence = random.choice(self.sentences)
            # if chosen_sentence not in self.random_summary:
            self.random_summary.append(chosen_sentence)

    def format_summary(self):
        # clean_summary_array = []
        # for sentence in self.random_summary:
        #     # sentence = ''.join(sentence[0].upper() + sentence[1:])
        #     clean_summary_array.append(sentence)
        # # clean_summary_array = ' '.join(clean_summary_array).replace(',', '.')
        self.random_summary = [''.join(sentence) for sentence in self.random_summary]
        self.random_summary = ' '.join(self.random_summary)
        return self.random_summary
