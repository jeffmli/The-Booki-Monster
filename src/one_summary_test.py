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
