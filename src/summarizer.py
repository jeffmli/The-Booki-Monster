import pandas as pd
import numpy as np
import re
from nltk import word_tokenize, sent_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from text_features import TextFeatures
import RAKE

class Summarizer(object):
    def __init__(self, documents, sentences,vectorizer, key_words, topic, length):
        '''
        Note: the "documents" coming in are actually full sentences while "sentences" are tokenized sentences.

        The vectorizer thats coming in is currently a CountVectorizer.
        '''
        self.documents = documents
        self.sentences = sentences
        self.vectorizer = vectorizer
        self.key_words = key_words
        # self.score = []
        self.sentence_scores = []
        self.topic = topic
        self.summary_array = np.array([])
        self.sentence_idx = np.array([])
        self.length = length

    '''
    -------------------------------Feature Engineering-------------------------------
    '''
    def tfidf_vectorizer(self):
        '''
        OUTPUT: term-frequency inverse document frequency object, term-frequency inverse-document frequency vectorize object
        '''
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf = tf_vectorizer.fit_transform(keyword_tokenize(self.documents))
        tfidf_feature_names = idf_vectorizer.get_feature_names()
        return tfidf, tfidf_vectorizer

    def count_vectorizer(self):
        '''
        OUTPUT: Count vectorizer fitted object, Count vectorizer object
        '''
        count_vectorizer = CountVectorizer(stop_words='english')
        count = tf_vectorizer.fit_transform(keyword_tokenize(self.documents))
        count_feature_names = idf_vectorizer.get_feature_names()
        return count, count_vectorizer

    def get_sentence_scores(self):
        '''
        Calculate sentence scores from engineered features
        '''
        text_features = TextFeatures()
        for position, sentence in enumerate(self.sentences):
            score = []
            '''
            Position-Weight Feature
            '''
            pos_percent = position/float(len(self.sentences))
            pos_weight = text_features.sentence_position_weight(pos_percent)
            '''
            Num tokens Feature
            - Divide the sentence score by the number of tokens in the sentence
            '''
            if len(sentence) != 0:
                num_words = len(sentence)
            else:
                num_words = 1
            '''
            Term-Frequency Feature
            '''
            term_freq_score = text_features.term_frequencies(sentence, self.topic, self.key_words)
            '''
            Add Named-Entity Feature
            - Upweight the score of the sentence if there's a Named-Entity.
            '''

            '''
            Add Verb Presence
            '''
            verb_count = text_features.presence_of_verb(sentence)
            '''
            Create sentence-feature vector
            '''
            # score.append(pos_weight)
            score.append(term_freq_score/num_words)
            score.append(verb_count)

            self.sentence_scores.append(pos_weight * sum(score))
        return self.sentence_scores

    '''
    -------------------------------Summarizing-------------------------------
    '''
    def summarize(self):
        '''
        Create sentence rank. Pick highest scoring sentence(s) to form summary.
        '''
        # how do I pick the top sentences and then maintain the order?
        self.sentences = np.array(self.sentences)

        sort_idx = np.argsort(self.sentence_scores)[::-1]
        # cumulative_importance = np.cumsum(self.sentence_scores[sort_idx]/float(np.sum(self.sentence_scores)))
        # # how should I weight the sentence that contains the key word?
        #
        # # cumulative_importance = np.cumsum(sentence_scores[sort_idx]/float(np.sum(sentence_scores)))
        #
        # top_n = np.where(cumulative_importance > 0.9)
        # important_sentence_idx = sort_idx[top_n]
        # sentence_idx = np.sort(important_sentence_idx)
        sentence_idx = sort_idx[:self.length]

        self.documents = np.array(self.documents)
        summary_array = self.documents[sentence_idx]
        self.summary_array = [''.join(sentence) for sentence in summary_array]


    def format_summary(self):
        '''
        Formatting the summary into a readable format.
        '''
        return ' '.join(self.summary_array)
