'''
Taking in a txt file and formatting the txt file in a way that can be fed into my model

Different Ideas:
- Split text file by chapters
- Split text file by word percentages

'''
import pandas as pd
import numpy as np
import re
from nltk import sent_tokenize

def load_data(filename):
    '''
    Load data
    '''
    f = open(filename)
    return f.read()

def get_rid_of_weird_characters(book):
    book = re.sub('[^A-Za-z0-9,;.?!'']+', ' ', book)
    return book

def get_sections(book):
    return book.split('\n\n\n\n\n\n')

def slice_introduction(book):
    '''
    Ideas:
    - Write a function that detects whether a section is the actual content or not.
    - Find a marker that marks the first chapter or introduction and slice everything
    after that.
    -
    '''
    pass

def chapter_paragraph_tag(book):
    '''
    Tags the beginning of each chapter and beginning and end of every paragraph
    - Create a chapter indicators list
    '''
    chapter_book = []
    chapter_indicators = ["Chapter", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN", "ELEVEN", "TWELVE", "THIRTEEN"]

    for section in book:
        section_split = section.split()
        if any(ind in section_split[:2] for ind in chapter_indicators):
            section = " <CHAPTERBEGIN> " + section
        else:
            section = " <PARAGRAPH> " + section
        chapter_book.append(section)
    return chapter_book

def combine_strings_split_on_chapter(book):
    '''
    Combine tagged strings into one and split on chapter tag
    '''
    book = " ".join(book)
    return book.split("<CHAPTERBEGIN>")

def split_by_section(book):
    '''
    For each chapter, split on paragraph tags. append these into paragraphs of chapter list.
    Append par_of_chapter list into book_list
    '''
    book_list = []
    for chapter in book:
        split_chapter = chapter.split("<PARAGRAPH>")
        paragraph_by_chapter = [s_chapter for s_chapter in split_chapter]
        book_list.append(paragraph_by_chapter)
    return book_list

def format_sentences(book):
    '''
    Tokenize all sentences in each chapter. Create a new data structure with the sentences tokenized
    '''
    book_list = []
    paragraph_list = []

    for chapter in book:
        chapter_list = []
        for paragraph in chapter:
            paragraph_tok = sent_tokenize(paragraph)
            chapter_list.append(paragraph_tok)
        book_list.append(chapter_list)
    return book_list

def get_



if __name__ == '__main__':
    # MoonwalkingwithEinstein_fullbook_cleaned.txt
    dirty_book = load_data('booktxt/MoonwalkingwithEinstein_fullbook_cleaned.txt')

    sliced_book = get_sections(dirty_book)
    kinda_clean_book = [get_rid_of_weird_characters(section) for section in sliced_book]
    more_clean_book = chapter_paragraph_tag(kinda_clean_book)
    combined = combine_strings_split_on_chapter(more_clean_book) #A list of chapters.
    split = split_by_section(combined)

    formatted_sentence = format_sentences(split)
