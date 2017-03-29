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
import ast

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

def aggregate_summary(documents, topic):
    '''
    Aggregate all chapter summaries into one total summary.
    '''
    book = []
    rand_book = []
    for chapter in documents:
        joined_chapter = " ".join([" ".join(string) for string in chapter])
        # You can't pass in all the tokens as sentences, that doesn't make sense.
        sentences = sent_tokenize(joined_chapter)
        sentences_tok = [word_tokenize(sentence) for sentence in sentences]
        key_words, tf, tf_feature_names, tf_vectorizer = get_key_words_vector(sentences)
        summary = generate_summary(sentences, sentences_tok, tf_vectorizer, key_words, topic = topic)
        book.append(summary)

        chapter_summary_object = RandomSummarizer(chapter, sentences, summary_length = len(sent_tokenize(summary))) # The length is the number of sentences in the chapter summ of summary.
        chapter_summary_object.sentence_generator()
        chapter_rand_summary = chapter_summary_object.format_summary()
        rand_book.append(chapter_rand_summary)
    return book, rand_book

def generate_summary(formatted_sentences, sentences_tok, tf_vectorizer, key_words, topic):
    '''
    Aggregate all functions created to generate summaries to generate summary.
    '''

    summary_object = Summarizer(formatted_sentences, sentences_tok, tf_vectorizer, key_words, topic)
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

def doc2vec_total(documents):

    '''
    To Do:
    - Add a length so it matches the length of the reference summaries.
    '''
    book = []
    for chapter in documents:
        chapter_summary = doc2vec(chapter)
        book.append(chapter_summary)
    return book

def doc2vec(chapter):
    joined_chapter = " ".join([" ".join(string) for string in chapter])
    tokenized_chapter = sent_tokenize(joined_chapter)

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
    similar_sentence_vectors = np.array(model.docvecs.most_similar('CHAP'))

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

    df['LDA Score'] = 0
    df['Doc2Vec Score'] = 0
    return df[:5]

def feature_build_summary(df, topic):
    '''
    Note: Remember to add the models in here when finished building them.
    '''
    df['LDA Summary'] = 0
    df['Doc2Vec Summary'] = 0
    df['Random Book'] = 0

    def agg(x):
        lda_book, rand_book = aggregate_summary(x, topic = 7)
        return lda_book

    def rand(x):
        lda_book, rand_book = aggregate_summary(x, topic = 7)
        return rand_book

    df['LDA Summary'] = df['book'].apply(lambda x : agg(x))
    df['Doc2Vec Summary'] = df['book'].apply(lambda x : doc2vec_total(x))
    df['Random Summary'] = df['book'].apply(lambda x : rand(x))

    def format(x):
        new = []
        for summary in x:
            for s in summary:
                new.append(s)
        return ''.join(new)

    df['LDA Summary'] = df['LDA Summary'].apply(lambda x : format(x))
    df['Doc2Vec Summary'] = df['Doc2Vec Summary'].apply(lambda x : format(x))
    df['Random Summary'] = df['Random Summary'].apply(lambda x : format(x))
    return df

def fill_df_with_scores(df):
    # ' '.join([str(s) for s in summary for summary in df['LDA Summary'].values])
    # df['LDA Score'] = rouge_score((' '.join(str(s) for s in summary for summary in df['LDA Summary'].values)), ' '.join(df['summary'].values), n = 2)
    # df['Doc2Vec Score'] = rouge_score((' '.join([str(s) for s in summary for summary in df['Doc2Vec Summary'].values])), ' '.join(df['summary'].values), n=2)
    # df['Random Score'] = rouge_score((' '.join([str(s) for s in summary for summary in df['Random Summary'].values])), ' '.join(df['summary'].values), n = 2)

    df['LDA Score'] = df['LDA Summary'].apply(lambda x : rouge_score(x, ' '.join(df['summary'].values)))
    df['Doc2Vec Score'] = df['Doc2Vec Summary'].apply(lambda x : rouge_score(x, ' '.join(df['summary'].values)))
    df['Random Summary Score'] = df['Random Summary'].apply(lambda x : rouge_score(x, ' '.join(df['summary'].values)))

    return df



if __name__=='__main__':
    '''
    ---------------- Load JSON file ----------------
    '''
    sum_book_dict_clean = import_data('books.json')
    # sum_book_dict = feature_build_summary(sum_book_dict_clean, topic = 7)
    df = create_dataframe_for_scores(sum_book_dict_clean)
    df = feature_build_summary(df, topic = 7)
    df = fill_df_with_scores(df)
    #
    # '''
    # ---------------- Load & Clean Data ------------------------
    # '''
    # dirty_book = load_data('booktxt/Moonwalking With Einstein.txt')
    # sliced_book = format_books.get_sections(dirty_book)
    # kinda_clean_book = [format_books.get_rid_of_weird_characters(section) for section in sliced_book]
    # more_clean_book = format_books.chapter_paragraph_tag(kinda_clean_book)
    # combined = format_books.combine_strings_split_on_chapter(more_clean_book) #A list of chapters.
    # split = format_books.split_by_section(combined)
    # formatted_sentence = format_books.format_sentences(split) # Won't work yet, will need to loop through chapters
    #
    # '''
    # ---------------- Featurize + Build Summary ------------------------
    # '''
    # lda_book, rand_book = aggregate_summary(formatted_sentence, topic=1)
    #
    # doc2vec_book = doc2vec_total(formatted_sentence)
    # '''
    # ---------------- Format Summaries for Scoring------------------------
    # '''
    # dirty_ref_summary = load_data('blinkistsummarytxt/blinkistsapiens.txt')
    # reference_summary_cleaned = clean_line(dirty_ref_summary)
    # reference_summary = format_summary(reference_summary_cleaned)
    #
    # joined_lda_book = ' '.join(lda_book)
    #
    # joined_rand_book =  [' '.join(chapter) for chapter in rand_book]
    # joined_random_book = ' '.join(joined_rand_book)
    #
    # joined_doc2vec = ' '.join(doc2vec_book)
    #
    # '''
    # ---------------- Scoring ------------------------
    # '''
    #
    # lda_score = rouge_score(joined_lda_book, reference_summary, n = 2)
    # doc2vec_score = rouge_score(joined_doc2vec, reference_summary, n=2)
    # random_score = rouge_score(joined_random_book, reference_summary, n = 2)
    #
    # lda_percentage_reduced = (len(' '.join(kinda_clean_book)) - len(joined_lda_book))/float(len(' '.join(kinda_clean_book)))
    # doc2vec_percent_reduced = (len(' '.join(kinda_clean_book)) - len(joined_doc2vec))/float(len(' '.join(kinda_clean_book)))
    # ref_reduced = (len(' '.join(kinda_clean_book))- len(reference_summary))/float(len(' '.join(kinda_clean_book)))
    #
    # '''
    # ---------------- Print ------------------------
    # '''
    # print 'Rouge Score for LDA Model :{0}'.format(lda_score)
    # print 'Rouge Score for Doc2Vec Model :{0}'.format(doc2vec_score)
    # print 'Rouge Score for Baseline: {0}'.format(random_score)
    # print 'Percentage LDA Reduced: {0}'.format(lda_percentage_reduced)
    # print 'Percentage Doc2Vec Reduced: {0}'.format(doc2vec_percent_reduced)
    # print 'Reference Summary Amount Reduced: {0}'.format(ref_reduced)
