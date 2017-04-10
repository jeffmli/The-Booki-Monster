import numpy as np
import pandas as pd
import glob

def read_full_book_txt(path):
    '''
    INPUT: Book text path
    OUTPUT: Cleaned Book dictionary
    '''
    book_dict = {}
    for filename in glob.glob(path):
        f = open(filename, 'r')
        book = f.read()
        book_dict[str(filename[24:-4])] = {'book': book}
    return book_dict

def read_summary_txt(path, book_dict):
    '''
    Open all the summary text files in Katrina Pulled Summary txt
    INPUT: Path, Book dictionary
    OUTPUT: Book dictionary with summaries
    '''
    titles = book_dict.keys()
    book_sum_dict = {}
    for filename in glob.glob(path):
        if filename[25:-4] in titles:
            f = open(filename, 'r')
            summary = f.read()
            book = book_dict[str(filename[25:-4])]['book']
            book_sum_dict[str(filename[25:-4])] = {'book': book, 'summary': summary}
    return book_sum_dict

if __name__=='__main__':
    book_dict = read_full_book_txt('Katrina Pulled Book txt/*.txt')
    book_sum_dict = read_summary_txt('Katrina Pulled Summaries/*.txt', book_dict)
