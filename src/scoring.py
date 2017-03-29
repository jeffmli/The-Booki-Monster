from nltk import word_tokenize


def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])


def rouge_score(my_summary, reference_summary, n = 2):
    my_summary_tokenized = word_tokenize(my_summary)
    ref_summary_tokenized = word_tokenize(reference_summary)
    my_ngrams = find_ngrams(my_summary_tokenized,n)
    ref_ngrams = find_ngrams(ref_summary_tokenized,n)
    both_ngrams = [ngram for ngram in my_ngrams if ngram in ref_ngrams]
    return len(both_ngrams)/float(len(ref_ngrams))
