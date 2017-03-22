import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from summarizer import Summarizer
from scoring import find_ngrams, rouge_score

def load_data(filename):
    f = open(filename)
    return f.read()

def split_by_period(chapter1):
    return chapter1.split('.')

def clean_line(chapter1):
    chapter1_sentences_cleaned = []
    for sentence in chapter1:
        sentence = sentence.strip('\n')
        sentence = re.sub('[^A-Za-z0-9]+', ' ', sentence)
        chapter1_sentences_cleaned.append(sentence)
    return chapter1_sentences_cleaned

def print_top_words(model, feature_names, n_top_words):
    key_words = {}
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        key_words[topic_idx] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
    print()
    return key_words


def create_dataframe(chapter1):
    df = pd.DataFrame({'Sentences': chapter1})
    return df

def format_sentences(df):
    return [row for row in df['tokenized_sents']]

def word_tokenize(df):
    df['tokenized_sents'] = df.apply(lambda row: nltk.word_tokenize(row['Sentences'].lower()), axis=1)
    df1 = df['tokenized_sents']
    df = df[df1.apply(lambda x:len(x)>0)]
    return df

def create_bag_of_words(df):
    bagofwords = list(set(np.concatenate(df.tokenized_sents)))
    return bagofwords

def vectorize(documents):
    tf_vectorizer = CountVectorizer(stop_words='english')
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()
    return tf, tf_feature_names, tf_vectorizer

def model(tf, k_topics):
    lda = LatentDirichletAllocation(n_topics=10, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
    fitted_lda = lda.fit(tf)
    return fitted_lda
    # How do I get the words that belong to n_topics?

def format_summary(summary_list):
    clean_summary_array = []
    for sentence in summary_list:
        sentence = sentence + "."
        clean_summary_array.append(sentence)
    clean_summary_array = ' '.join(clean_summary_array).replace(',', '.')
    return clean_summary_array

if __name__=='__main__':
    '''
    ---------------- Load & Clean Data ------------------------
    '''
    chapter1 = load_data('test_sapiens_chapter_1.txt')
    chapter1_sentences = split_by_period(chapter1)
    chapter1_sentences_cleaned = clean_line(chapter1_sentences)
    df = create_dataframe(chapter1_sentences_cleaned)
    df = word_tokenize(df)
    sentences = format_sentences(df)
    '''
    ---------------- Featurize/Run Model ------------------------
    '''
    bagofwords = create_bag_of_words(df)
    tf, tf_feature_names, tf_vectorizer = vectorize(chapter1_sentences_cleaned)
    fitted_lda = model(tf, 10)

    key_words = print_top_words(fitted_lda, tf_feature_names, n_top_words = 50)

    '''
    ---------------- Summarizer ------------------------
    '''
    summary_object = Summarizer(bagofwords, sentences, tf_vectorizer, key_words, 1)
    # summary_sent = summary.get_sentence_scores()
    summary_object.get_sentence_scores()
    summary_object.summarize()
    summary = summary_object.format_summary()
    print summary
    '''
    ---------------- Scoring ------------------------
    - Note, will write a function to pull random sentences to create a baseline
    for scoring.
    '''
    dirty_ref_summary = load_data('blinkistsummarytxt/blinkistsapiens.txt')
    split_reference_summary = split_by_period(dirty_ref_summary)
    reference_summary_cleaned = clean_line(split_reference_summary)
    reference_summary = format_summary(reference_summary_cleaned)

    score = rouge_score(summary, reference_summary, n = 2)
    print score
