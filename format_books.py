'''
Taking in a txt file and formatting the txt file in a way that can be fed into my model

Different Ideas:
- Split text file by chapters
- Split text file by word percentages

'''
import pandas as pd
import numpy as np
import re

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

if __name__ == '__main__':
    dirty_book = load_data('booktxt/Mindset_fullbook_cleaned.txt')

    sliced_book = get_sections(dirty_book)
    kinda_clean_book = [get_rid_of_weird_characters(section) for section in sliced_book]
