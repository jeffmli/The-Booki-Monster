import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from nltk import sent_tokenize, word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
import sys
import pickle


def get_user_input():
    print "Please enter your text to be eaten by the Booki Monster"
    text = raw_input()
    print "How long do you want your summary to be?"
    length = raw_input()
    return text, length


def doc2vec(text, length):
    '''
    Note, figure out a way to fix this regex.
    '''

    text = re.sub('[^A-Za-z0-9,;.?!]+', ' ', text)
    text_tokenized = sent_tokenize(text)

    class LabeledLineSentence(object):
        '''
        Create generator of sentences & labels to create Doc2Vec model
        '''
        def __init__(self, x):
            self.x = x
        def __iter__(self):
            yield LabeledSentence(words=word_tokenize(text), tags=['TEXT'])
            for uid, line in enumerate(self.x):
                yield LabeledSentence(words=line.split(), tags=[int(uid)])

    x = LabeledLineSentence(text_tokenized)
    model = Doc2Vec(min_count = 1)
    model.build_vocab(x)
    model.train(x)
    similar_sentence_vectors = model.docvecs.most_similar('TEXT', topn = length)

    return text_tokenized,similar_sentence_vectors

def combine_into_summary(text_tokenized, similar_sentence_vectors):
    vector_index = sorted([int(vector[0]) for vector in similar_sentence_vectors])
    chapter_summary = [text_tokenized[index] for index in vector_index]
    chapter_summary = ' '.join(chapter_summary)
    return chapter_summary


if __name__=='__main__':
    print "Please paste your text here for summarization."
    text = sys.stdin.read()
    length = raw_input("How long do you want your summary to be?\n")
    length = int(length)
    text_tokenized,similar_sentence_vectors =  doc2vec(text, length)
    summary = combine_into_summary(text_tokenized, similar_sentence_vectors)

    afile = open('C:\d.pkl', 'wb')
    pickle.dump(super_dict, afile)
    afile.close()
