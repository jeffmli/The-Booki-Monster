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
        self.score = 0.0
        self.sentence_scores = []
        self.topic = topic
        self.summary_array = np.array([])
        self.sentence_idx = np.array([])
        self.length = length

    '''
    -------------------------------Feature Engineering-------------------------------
    - Apply RAKE to pull multiple word tokens.
        Note: The sentences coming in are already tokenized
    - Create tfidf, count, doc2vec, lda2vec vectors of sentences in book.
    - Score sentence vectors by computing against vector of entire book.
    - Rank sentences based off sentence vector scores.
    - Assign specific sentence weights to the sentence vector scores.
    - Re-rank sentences based on assigned weights.
    '''
    def tfidf_vectorizer(self):
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf = tf_vectorizer.fit_transform(keyword_tokenize(self.documents))
        tfidf_feature_names = idf_vectorizer.get_feature_names()
        return tfidf, tfidf_vectorizer

    def count_vectorizer(self):
        count_vectorizer = CountVectorizer(stop_words='english')
        count = tf_vectorizer.fit_transform(keyword_tokenize(self.documents))
        count_feature_names = idf_vectorizer.get_feature_names()
        return count, count_vectorizer

    def lda_vector(self):
        pass

    def get_sentence_scores(self):
        '''
        Calculate sentence scores from engineered features
        '''
        text_features = TextFeatures()
        for sentence in self.sentences:
            # '''
            # Position-Weight Feature
            # '''
            # position_dict = text_features.sentence_position_weight(self.documents, self.sentences)
            # '''
            # Num Words Feature
            # '''
            # num_words = len(sentence)
            # '''
            # Term-Frequency Feature
            # '''
            term_freq_score = text_features.term_frequencies(sentence, self.topic, self.key_words)
            self.sentence_scores.append(term_freq_score)
        return self.sentence_scores

    '''
    -------------------------------Summarizing-------------------------------
    '''




    def summarize(self):
        '''
        Create sentence rank. Pick highest scoring sentence(s) to form summary.
        Note: Divide out sentence rank into another function if I have more time.

        - Rake boosted ROUGE-N score by about 2%

        - Preprocessing
        - keyword extractor
        - generate candidate keywords--> fed to LDA vectors
        - Google ngram
        '''
        # how do I pick the top sentences and then maintain the order?
        self.sentence_scores = np.array(self.sentence_scores)
        self.sentences = np.array(self.sentences)

        sort_idx = np.argsort(self.sentence_scores)[::-1]
        cumulative_importance = np.cumsum(self.sentence_scores[sort_idx]/float(np.sum(self.sentence_scores)))
        # how should I weight the sentence that contains the key word?

        # cumulative_importance = np.cumsum(sentence_scores[sort_idx]/float(np.sum(sentence_scores)))

        top_n = np.where(cumulative_importance > 0.9)
        important_sentence_idx = sort_idx[top_n]
        sentence_idx = np.sort(important_sentence_idx)
        sentence_idx = sentence_idx[:self.length]

        self.documents = np.array(self.documents)
        summary_array = self.documents[sentence_idx]
        self.summary_array = [''.join(sentence) for sentence in summary_array]


    def format_summary(self):
        '''
        Formatting the summary into a readable format.
        '''
        return ' '.join(self.summary_array)
