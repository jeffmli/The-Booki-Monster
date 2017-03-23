import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from summarizer import Summarizer
from scoring import find_ngrams, rouge_score
from RandomSummarizer import RandomSummarizer

def load_data(filename):
    '''
    Load data
    '''
    f = open(filename)
    return f.read()

def clean_line(chapter1):
    chapter1 = chapter1.split('.')
    chapter1_sentences_cleaned = []
    for sentence in chapter1:
        sentence = sentence.strip('\n')
        sentence = re.sub('[^A-Za-z0-9]+', ' ', sentence)
        chapter1_sentences_cleaned.append(sentence)
    return chapter1_sentences_cleaned

def create_dataframe(chapter1):
    return pd.DataFrame({'Sentences': chapter1})

def word_tokenize(df):
    df['tokenized_sents'] = df.apply(lambda row: nltk.word_tokenize(row['Sentences'].lower()), axis=1)
    df1 = df['tokenized_sents']
    df = df[df1.apply(lambda x:len(x)>0)]
    return [row for row in df['tokenized_sents']]

def vectorize(documents):
    tf_vectorizer = CountVectorizer(stop_words='english')
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()
    return tf, tf_feature_names, tf_vectorizer

def latent_dirichlet_allocation(tf, k_topics):
    lda = LatentDirichletAllocation(n_topics=10, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
    fitted_lda = lda.fit(tf)
    return fitted_lda

def get_top_words(model, feature_names, n_top_words):
    key_words = {}
    for topic_idx, topic in enumerate(model.components_):
        # print("Topic #%d:" % topic_idx)
        " ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]])
        key_words[topic_idx] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
    # print()
    return key_words

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
    chapter1_sentences_cleaned = clean_line(chapter1)
    df = create_dataframe(chapter1_sentences_cleaned)
    sentences = word_tokenize(df)
    '''
    ---------------- Featurize/Run Model ------------------------
    '''
    tf, tf_feature_names, tf_vectorizer = vectorize(chapter1_sentences_cleaned)
    fitted_lda = latent_dirichlet_allocation(tf, 10)

    key_words = get_top_words(fitted_lda, tf_feature_names, n_top_words = 50)
    '''
    ---------------- Feature Engineering ------------------------
    '''
    summary_object = Summarizer(chapter1_sentences_cleaned, sentences, tf_vectorizer, key_words, 6)
    summary_object.get_sentence_scores()
    '''
    ---------------- Summarizer ------------------------
    '''

    summary_object.summarize()
    summary, summary_array = summary_object.format_summary()
    '''
    ---------------- Random Sentence Summary ------------------------
    '''
    random_summary_object = RandomSummarizer(chapter1_sentences_cleaned, sentences, summary_length = len(summary_array))
    random_summary_object.sentence_generator()
    random_summary = random_summary_object.format_summary()
    '''
    ---------------- Scoring ------------------------
    '''
    dirty_ref_summary = load_data('blinkistsummarytxt/blinkistsapiens.txt')
    reference_summary_cleaned = clean_line(dirty_ref_summary)
    reference_summary = format_summary(reference_summary_cleaned)

    score = rouge_score(summary, reference_summary, n = 2)
    random_score = rouge_score(random_summary, reference_summary, n = 2)

    percentage_reduced = (len(chapter1_sentences_cleaned) - len(summary_array))/float(len(chapter1_sentences_cleaned))
    '''
    ---------------- Print Results ------------------------
    '''

    print 'Rouge Score for Model :{0}'.format(score)
    print 'Rouge Score for Baseline: {0}'.format(random_score)
    print 'Percentage Reduced: {0}'.format(percentage_reduced)
