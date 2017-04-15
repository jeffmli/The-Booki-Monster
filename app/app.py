from flask import Flask, request, render_template
from flask import render_template
import os
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from app_summarizer import doc2vec, combine_into_summary
from django.utils.encoding import smart_str

app = Flask(__name__)

@app.route('/')
def index():
    '''
    This will render the homepage, introducing the Booki Monster
    '''
    return render_template("index.html")

@app.route('/summarizer', methods = ['GET','POST'])
def summarizer():
    '''
    This will render the page where the user can copy + paste their text to test model.
    '''
    if request.method == 'POST':
        text = request.form['text']
        return render_template('summarizer.html',result=request.form['text']), text
    if request.method == 'GET':
        return render_template('summarizer.html')

@app.route('/summary', methods = ['POST'])
def summary():
    '''
    This will return the summary of inputted text based on user input.
    '''
    text = request.form['text'].encode('utf8')
    length = request.form['length'].encode('utf8')
    length = int(length)
    text_tokenized,similar_sentence_vectors =  doc2vec(text, length=length)
    summary = combine_into_summary(text_tokenized, similar_sentence_vectors)
    return render_template("summary.html", summary = summary)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0',port=port,debug=True,threaded=True)
