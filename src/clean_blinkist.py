import pandas as pd
import numpy as np
import json
import re
import glob
import os
import format_books

def load_data(filename):
    with open('books.json') as books:
        d = json.load(books)
        return d

def get_titles(summaries):
    return [summaries[i]['title'] for i in range(len(summaries))]

def aggregate_summaries(summaries):
    '''
    Create a dictionary of each summary with the title as the key and all it's summary content as
    the values.
    '''
    summary_dict = {}
    for summary in summaries:
        text = []
        for key in summary['chapters']:
            text.append(key['text'])
        text = ' '.join(text)
        text = get_rid_of_tags(text)
        summary_dict[summary['title']] = text
    return summary_dict

def get_rid_of_tags(text):
    text = re.sub('[</p>]', '', text)
    text = text.replace('strong', '')
    text = re.sub('[^A-Za-z0-9|.|,|!|?|']+', ' ', text)
    return text

def read_all_files_in_folder(path, summary_dict):
    '''
    Open all the files in the folder.
    Check filename match with summary_dict title.
    Store in summary_dict with key = 'book' and book text as value.
    '''
    titles = [title.lower() for title in summary_dict.keys()]

    titles = summary_dict.keys()
    sum_book_dict = {}
    for filename in glob.glob(path):
        # filename = filename.replace(' ','_')
        if filename[16:-4] in titles:
            index = [i for i, s in enumerate(titles) if filename[16:-4] in s]
            f = open(filename, 'r')
            book = f.read()
            sum_book_dict[titles[index[0]]] = {'summary':summary_dict[titles[index[0]]], 'book':book}
    return sum_book_dict

def clean_book_text(titles, sum_book_dict):
    for book in sum_book_dict:
        whole_text = sum_book_dict[book]['book']

        sliced_book = format_books.get_sections(whole_text)
        kinda_clean_book = [format_books.get_rid_of_weird_characters(section) for section in sliced_book]
        more_clean_book = format_books.chapter_paragraph_tag(kinda_clean_book)
        combined = format_books.combine_strings_split_on_chapter(more_clean_book) #A list of chapters.
        split = format_books.split_by_section(combined)
        formatted_sentence = format_books.format_sentences(split)

        sum_book_dict[book]['book'] = formatted_sentence
    return sum_book_dict




if __name__=='__main__':
    summaries = load_data('books.json')
    titles = get_titles(summaries)
    summary_dict = aggregate_summaries(summaries)
    sum_book_dict_dirty = read_all_files_in_folder('blinkistbooktxt/*.txt', summary_dict)

    sum_book_dict_clean = clean_book_text(titles, sum_book_dict_dirty)
