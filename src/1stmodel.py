import pandas as pd
import numpy as np
import re
from nltk import sent_tokenize, word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from summarizer import Summarizer
from scoring import find_ngrams, rouge_score
from RandomSummarizer import RandomSummarizer
import format_books

def load_data(filename):
    '''
    Load data
    '''
    f = open(filename)
    return f.read()

def clean_line(chapter1):
    '''
    Get rid of useless characters.
    '''
    chapter1 = chapter1.split('.')
    chapter1_sentences_cleaned = []
    for sentence in chapter1:
        sentence = sentence.strip('\n')
        sentence = re.sub('[^A-Za-z0-9]+', ' ', sentence)
        chapter1_sentences_cleaned.append(sentence)
    return chapter1_sentences_cleaned

def create_dataframe(chapter1):
    '''
    Turn into a dataframe.
    '''
    return pd.DataFrame({'Sentences': chapter1})

def tokenize(df):
    '''
    Function that tokenizes the words of each sentence.
    '''
    df['tokenized_sents'] = df.apply(lambda row: word_tokenize(row['Sentences'].lower()), axis=1)
    df1 = df['tokenized_sents']
    df = df[df1.apply(lambda x:len(x)>0)]
    return [row for row in df['tokenized_sents']]

def vectorize(documents):
    '''
    Turning sentences into vectors
    '''
    tf_vectorizer = CountVectorizer(stop_words='english')
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()
    return tf, tf_feature_names, tf_vectorizer

def latent_dirichlet_allocation(tf, k_topics):
    '''
    LDA topic model.
    '''
    lda = LatentDirichletAllocation(n_topics=10, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
    fitted_lda = lda.fit(tf)
    return fitted_lda

def get_top_words(model, feature_names, n_top_words):
    '''
    using the LDA topic model to extract key words
    '''
    key_words = {}
    for topic_idx, topic in enumerate(model.components_):
        # print("Topic #%d:" % topic_idx)
        " ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]])
        key_words[topic_idx] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
    # print()
    return key_words

def format_summary(summary_list):
    '''
    Re-format summary.
    '''
    clean_summary_array = []
    for sentence in summary_list:
        sentence = sentence + "."
        clean_summary_array.append(sentence)
    clean_summary_array = ' '.join(clean_summary_array).replace(',', '.')
    return clean_summary_array

def load_clean_data(filename):
    '''
    Aggregate all functions created to load & clean data into one place.
    '''
    chapter1 = load_data(filename)
    chapter1_sentences_cleaned = clean_line(chapter1)
    df = create_dataframe(chapter1_sentences_cleaned)
    # sentences = sent_tokenize(chapter1_sentences_cleaned)
    return chapter1, chapter1_sentences_cleaned, df

def get_key_words_vector(documents):
    '''
    Get all key words using LDA model
    '''
    tf, tf_feature_names, tf_vectorizer = vectorize(documents)
    fitted_lda = latent_dirichlet_allocation(tf, 10)
    key_words = get_top_words(fitted_lda, tf_feature_names, n_top_words = 50)
    return key_words, tf, tf_feature_names, tf_vectorizer

def generate_summary(documents, sentences, tf_vectorizer, key_words, topic):
    '''
    Aggregate all functions created to generate summaries to generate summary.
    '''
    summary_object = Summarizer(documents, sentences, tf_vectorizer, key_words, topic)
    summary_object.get_sentence_scores()
    summary_object.summarize()
    summary, summary_array = summary_object.format_summary()
    return summary, summary_array

def aggregate_summary(documents, topic):
    '''
    Aggregate all chapter summaries into one total summary.
    '''
    book = []
    for chapter in documents:
        joined_chapter = " ".join([" ".join(string) for string in chapter])
        joined_chapter = word_tokenize(joined_chapter)
        key_words, tf, tf_feature_names, tf_vectorizer = get_key_words_vector(joined_chapter)
        summary, summary_array = generate_summary(documents, joined_chapter, tf_vectorizer, key_words, topic)
        book.append(summary)
    return book, joined_chapter

def random_summary(documents, sentences, topic):
    '''
    Create random summary from book.
    '''
    book = []
    for chapter in documents:
        joined_chapter = " ".join([" ".join(string) for string in chapter])
        chapter_summary_object = RandomSummarizer(chapter, sentences, summary_length = len(joined_chapter))
        chapter_summary_object.sentence_generator()
        chapter_rand_summary = chapter_summary_object.format_summary()
        book.append(chapter_rand_summary)
    return book

# def create_sentences(documents):
#     book = []
#     for chapter in documents:
#         joined_chapter = " ".join([" ".join(string) for string in chapter])



if __name__=='__main__':
    '''
    ---------------- Load & Clean Data ------------------------
    '''
    dirty_book = load_data('booktxt/MoonwalkingwithEinstein_fullbook_cleaned.txt')
    sliced_book = format_books.get_sections(dirty_book)
    kinda_clean_book = [format_books.get_rid_of_weird_characters(section) for section in sliced_book]
    more_clean_book = format_books.chapter_paragraph_tag(kinda_clean_book)
    combined = format_books.combine_strings_split_on_chapter(more_clean_book) #A list of chapters.
    split = format_books.split_by_section(combined)
    formatted_sentence = format_books.format_sentences(split) # Won't work yet, will need to loop through chapters
    # sentences = word_tokenize(formatted_sentence)

    # chapter1, chapter1_sentences_cleaned, df = load_clean_data('test_sapiens_chapter_1.txt')
    # sentences = word_tokenize(df)
    '''
    ---------------- Featurize + Build Summary ------------------------
    '''

    # key_words, tf, tf_feature_names, tf_vectorizer = get_key_words_vector(formatted_sentence)
    # summary, summary_array = generate_summary(formatted_sentence, sentences, tf_vectorizer, key_words, topic = 6)
    book, joined_chapter= aggregate_summary(formatted_sentence, topic=6)
    '''
    ---------------- Random Sentence Summary ------------------------
    '''
    # random_summary_object = RandomSummarizer(formatted_sentence, sentences, summary_length = len(summary_array))
    # random_summary_object.sentence_generator()
    # random_summary = random_summary_object.format_summary()
    rand_book = random_summary(formatted_sentence, joined_chapter, topic= 6)
    # '''
    # ---------------- Scoring ------------------------
    # '''
    # dirty_ref_summary = load_data('blinkistsummarytxt/blinkistsapiens.txt')
    # reference_summary_cleaned = clean_line(dirty_ref_summary)
    # reference_summary = format_summary(reference_summary_cleaned)
    #
    # score = rouge_score(summary, reference_summary, n = 2)
    # random_score = rouge_score(random_summary, reference_summary, n = 2)
    #
    # percentage_reduced = (len(chapter1_sentences_cleaned) - len(summary_array))/float(len(chapter1_sentences_cleaned))
    # '''
    # ---------------- Print Results ------------------------
    # '''
    #
    # print 'Rouge Score for Model :{0}'.format(score)
    # print 'Rouge Score for Baseline: {0}'.format(random_score)
    # print 'Percentage Reduced: {0}'.format(percentage_reduced)
