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
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
import clean_blinkist
from sklearn.metrics.pairwise import cosine_similarity
import rake
import pickle

def table_of_contents():
    '''
    1. Load data
    2. Clean line
    3. Format Summary
    4. Aggregate_summary
        a. generate_summary
        b. get_key_words_vector
            - Get Top Words
            - Latent Dirichlect Allocation
            - vectorize
    5. doc2vec total
        a. doc2vec
    '''
    pass

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

def aggregate_summary(documents, topic, length):
    '''
    Aggregate all section summaries into one summary.
    '''
    book = []
    rand_book = []
    for chapter in documents:
        joined_chapter = " ".join(["".join(string) for string in chapter]) # Do I need this? I don't think so.
        # You can't pass in all the tokens as sentences, that doesn't make sense.
        sentences = sent_tokenize(joined_chapter)
        sentences_tok = [keyword_tokenize(sentence) for sentence in sentences]
        key_words, tf, tf_feature_names, tf_vectorizer = get_key_words_vector(sentences)
        summary = generate_summary(sentences, sentences_tok, tf_vectorizer, key_words, topic = topic, length = length)
        book.append(summary)

        chapter_summary_object = RandomSummarizer(sentences, summary_length = length) # The length is the number of sentences in the chapter summ of summary.
        chapter_summary_object.sentence_generator()
        chapter_rand_summary = chapter_summary_object.format_summary()
        rand_book.append(chapter_rand_summary)
    return book, rand_book

def generate_summary(formatted_sentences, sentences_tok, tf_vectorizer, key_words, topic, length):
    '''
    Aggregate all functions created to generate summaries to generate summary.
    '''

    summary_object = Summarizer(formatted_sentences, sentences_tok, tf_vectorizer, key_words, topic, length)
    summary_object.get_sentence_scores()
    summary = summary_object.summarize()
    summary = summary_object.format_summary()

    return summary

def get_key_words_vector(documents):
    '''
    Get all key words using LDA model
    '''
    tf, tf_feature_names, tf_vectorizer = vectorize(documents)
    fitted_lda = latent_dirichlet_allocation(tf, 10)
    key_words = get_top_words(fitted_lda, tf_feature_names, n_top_words = 50)
    return key_words, tf, tf_feature_names, tf_vectorizer

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

def keyword_tokenize(text):
    stoppath = 'RAKE-tutorial/SmartStoplist.txt'
    rake_object = rake.Rake(stoppath, 5, 3, 4)
    sentenceList = rake.split_sentences(text)
    stopwordpattern = rake.build_stop_word_regex(stoppath)
    phraseList = rake.generate_candidate_keywords(sentenceList, stopwordpattern)
    wordscores = rake.calculate_word_scores(phraseList)
    return phraseList

def vectorize(documents):
    '''
    Turning sentences into multiple vectors:
    - Currently using just a Count Vectorizer to pass through LDA model
    - Add a whole book to vectorize the whole book
    '''

    '''
    Count-Vectorizer
    '''

    tf_vectorizer = CountVectorizer(stop_words='english', tokenizer = keyword_tokenize)
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()

    '''
    Tfidf-Vectorizer
    '''
    # idf_vectorizer = TfidfVectorizer(stop_words='english')
    # idf = tf_vectorizer.fit_transform(keyword_tokenize(documents))
    # idf_feature_names = idf_vectorizer.get_feature_names()

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

def doc2vec_total(documents, length):

    '''
    To Do:
    - Add a length so it matches the length of the reference summaries.
    '''
    book = []
    for chapter in documents:
        chapter_summary = doc2vec(chapter, length)
        book.append(chapter_summary)
    return book

def doc2vec(chapter, length):
    if type(chapter) == list:
        joined_chapter = " ".join(["".join(string) for string in chapter])
        tokenized_chapter = sent_tokenize(joined_chapter)
    else:
        tokenized_chapter = sent_tokenize(chapter)

    class LabeledLineSentence(object):
        '''
        Create generator of sentences & labels to create Doc2Vec model
        '''
        def __init__(self, x):
            self.x = x
        def __iter__(self):
            yield LabeledSentence(words=' '.join(self.x).split(), tags=['CHAP'])
            for uid, line in enumerate(self.x):
                yield LabeledSentence(words=line.split(), tags=[int(uid)])

    x = LabeledLineSentence(tokenized_chapter)

    model = Doc2Vec(min_count = 1)
    model.build_vocab(x)
    model.train(x)
    similar_sentence_vectors = np.array(model.docvecs.most_similar('CHAP', topn = length))

    vector_index = [int(vector[0]) for vector in similar_sentence_vectors]
    chapter_summary = [tokenized_chapter[index] for index in vector_index]
    chapter_summary = ' '.join(chapter_summary)
    return chapter_summary

def import_data(filename):
    summaries = clean_blinkist.load_data(filename)
    titles = clean_blinkist.get_titles(summaries)
    summary_dict = clean_blinkist.aggregate_summaries(summaries)
    sum_book_dict_dirty = clean_blinkist.read_all_files_in_folder('blinkistbooktxt/*.txt', summary_dict)
    sum_book_dict_clean = clean_blinkist.clean_book_text(titles, sum_book_dict_dirty)
    return sum_book_dict_clean


def create_dataframe_for_scores(sum_book_dict):
    '''
    Creating a dataframe to store all the scores calculated by the model for each book.
    '''

    df = pd.DataFrame(list(sum_book_dict.iteritems()), columns= ["Title", "Contents"])
    s = pd.DataFrame((d for idx, d in df['Contents'].iteritems()))
    df = df.join(s)
    return df[11:12]

def feature_build_summary(df, topic):
    '''
    Note: Remember to add the models in here when finished building them.
    '''
    def agg(x):
        lda_book, rand_book = aggregate_summary(x, topic = 7, length = 11)
        return lda_book

    def rand(x):
        lda_book, rand_book = aggregate_summary(x, topic = 7, length = 11)
        return rand_book

    df['LDA Summary'] = df['book_split_n'].apply(lambda x : agg(x))
    df['Doc2Vec Summary'] = df['book_split_n'].apply(lambda x : doc2vec_total(x, length = 11))
    df['Random Summary'] = df['book_split_n'].apply(lambda x : rand(x))

    def format_(x):
        new = []
        for summary in x:
            for s in summary:
                new.append(s)
        return ''.join(new)

    df['LDA Summary'] = df['LDA Summary'].apply(lambda x : format_(x))
    df['Doc2Vec Summary'] = df['Doc2Vec Summary'].apply(lambda x : format_(x))
    df['Random Summary'] = df['Random Summary'].apply(lambda x : format_(x))
    return df

def build_max_summ_score(df):

    def doc2vec_max_sum(ref_summary, book):
#     joined_chapter = " ".join([" ".join(string) for string in chapter])
#     tokenized_chapter = sent_tokenize(joined_chapter)
        class LabeledLineSentence(object):
            '''
            Create generator of reference summary & entire book to create vectors
            '''
            def __init__(self, summary, book):
                self.summary = summary
                self.book = book
            def __iter__(self):
                yield LabeledSentence(words= ' '.join(self.summary).split(), tags=['REF SUMMARY'])
                yield LabeledSentence(words = ' '.join(self.book).split(), tags = ['WHOLE BOOK'])
                for uid, line in enumerate(self.book):
                    yield LabeledSentence(words=line.split(), tags=[int(uid)])


        x = LabeledLineSentence(ref_summary, book)

        model = Doc2Vec()
        model.build_vocab(x)
        model.train(x)
        similar_sentences = np.array(model.docvecs.most_similar('REF SUMMARY', topn = 229))

        vector_index = [int(vector[0]) for vector in similar_sentences[1:]]
        sim_sentence = [book[index] for index in vector_index[:100]]
        total_sentence = ' '.join(sim_sentence)
        return total_sentence

    def f(x):
        return rouge_score[x[0], x[1]]

    df['Reference Produced Summary'] = df[['summary', 'all_sentences']].apply(lambda x: doc2vec_max_sum(x[0], x[1]),axis = 1)
    df['Max Rouge Score'] = df[['Reference Produced Summary', 'summary']].apply(lambda x: rouge_score(x[0], x[1]), axis = 1)
    return df


def fill_df_with_scores(df):
    df['summary'] = df['summary'].apply(lambda x: ''.join(x))
    df['LDA Split Score'] = df[['LDA Summary', 'summary']].apply(lambda x : rouge_score(x[0], x[1]), axis = 1)
    df['Doc2Vec Split Score'] = df[['Doc2Vec Summary', 'summary']].apply(lambda x : rouge_score(x[0], x[1]), axis = 1)
    df['Random Split Summary Score'] = df[['Random Summary', 'summary']].apply(lambda x : rouge_score(x[0], x[1]), axis = 1)
    df['LDA Full Book Score'] = df[['LDA Full Book Summary', 'summary']].apply(lambda x : rouge_score(x[0], x[1]), axis = 1)
    df['Doc2Vec Full Book Score'] = df[['Doc2Vec Full Book Summary', 'summary']].apply(lambda x : rouge_score(x[0], x[1]), axis = 1)
    df['Full Book Random Summary Score'] = df[['Full Book Random Summary', 'summary']].apply(lambda x : rouge_score(x[0], x[1]), axis = 1)
    return df

def add_split_percent_book(df, n = 10):
    '''
    Add new column that splits the book by n times and an entire book of all it's sentences.
    '''
    # df['all_sentences'] = df['book'].apply(lambda x: sent_tokenize(x))

    def split_book(x):
        return format_books.split_by_percentage(x, n = n)

    df['all_sentences'] = df['book'].apply(lambda x: sent_tokenize(x))
    df['book_split_n'] = df['all_sentences'].apply(lambda x: split_book(x))

    return df

def whole_book_lda(df, topic = 7):

    def lda_summary(x):
        sentences = sent_tokenize(x)
        sentences_tok = [word_tokenize(sentence) for sentence in sentences]
        key_words, tf, tf_feature_names, tf_vectorizer = get_key_words_vector(sentences)
        summary = generate_summary(sentences, sentences_tok, tf_vectorizer, key_words, topic = topic, length = 104)

        rand_summary_object = RandomSummarizer(sentences, summary_length = len(sent_tokenize(summary))) # The length is the number of sentences in the chapter summ of summary.
        rand_summary_object.sentence_generator()
        rand_summary = rand_summary_object.format_summary()
        return summary, key_words

    def rand_summary(x):
        sentences = sent_tokenize(x)
        sentences_tok = [word_tokenize(sentence) for sentence in sentences]
        key_words, tf, tf_feature_names, tf_vectorizer = get_key_words_vector(sentences)
        summary = generate_summary(sentences, sentences_tok, tf_vectorizer, key_words, topic = topic, length = 104)

        rand_summary_object = RandomSummarizer(sentences, summary_length = len(sent_tokenize(summary))) # The length is the number of sentences in the chapter summ of summary.
        rand_summary_object.sentence_generator()
        rand_summary = rand_summary_object.format_summary()
        rand_summary = ''.join(rand_summary)
        return rand_summary

    df['LDA Full Book Summary'] = df['book'].apply(lambda x : lda_summary(x))
    df['Full Book Random Summary'] = df['book'].apply(lambda x : rand_summary(x))

    return df

def whole_book_doc2vec(df):
    df['Doc2Vec Full Book Summary'] = df['book'].apply(lambda x : doc2vec(x, length = 100))
    return df

if __name__=='__main__':
    # sum_book_dict_clean = import_data('books.json')

    file2 = open('C:\d.pkl', 'r')
    sum_book_dict = pickle.load(file2)
    file2.close()

    df = create_dataframe_for_scores(sum_book_dict)
    df = add_split_percent_book(df, n = 10)
    df = feature_build_summary(df, topic = 6)
    df = whole_book_lda(df)
    df = whole_book_doc2vec(df)
    df = fill_df_with_scores(df)
    df = build_max_summ_score(df)
