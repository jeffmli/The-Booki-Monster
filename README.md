# The Booki Monster
![alt tag](http://www.babydigezt.com/wp-content/uploads/2014/09/book-monster.jpg)

One of the biggest problems I've noticed since entering the "real world", is that it's hard to find time for things. Going on adventures, finding a girlfriend, building new social groups, having interesting hobbies...... and reading books. Or maybe, you're just too hungover....... 

But a book that might take 5-10 hours to read? Well, what if I told you that you could get the key points from the book without actually reading the book? I want to introduce you to the "Booki Monster." A machine-learning powered monster that reads your books and summarizes them for you. 

My goal with this README, is for a non-technical person to understand technically how I built my project. 

# 1.Who is the Booki Monster?

You see those pile of books on your drawer that you're "too busy" to crack open? Well, that's the Booki Monster's food. Feed the Booki Monster your books and then she'll spit out the golden nuggets in the form of summaries.

# 2.Feeding the Monster

You know that feeling when you're with your friends, you want to eat out, but can't decide where to eat because there are wayyyy too many options? Feeding the Booki Monster was the same, I had too many options: science fiction, business, self-help, psychology, scientific research etc. 

And for those who understand product marketing, when you're product is for everyone, it's for no one.  I'd prefer to make the Booki Monster generate high-quality summaries for a more niche, targeted set of books, than mediocre summaries for many books. 

So I settled on feeding the Booki Monster only business books for this reason. Plus, Blinkist.com, a company that produces human-written summaries, happily agreed to send over their human-written summaries, so I can measure the Booki Monster summaries quality. 

With this understanding, grab your nearest surgeon, and let's start dissecting the body of the Booki Monster. Mmmmm.... tasty........

# 3.The Booki Monster's Body (Technical)

# Method

When creating the Booki Monster, I had a couple different options:

1. Sentence Extraction: It's similar to DJ'ing vs. Music Production. Am I using the songs already created? Or am I creating new sounds? Sentence Extraction is like DJ'ing, taking the text already written and using them as the summary. 
2. Abstractive Methods: Abstractive Methods are kinda like creating the sounds yourself. In the context of summarizing, it means that the machine needs to understand the text on a much, much deeper level.
3. Graph-Based: Graph-Based is more like DJ'ing than Music Production. Imagine all your Facebook friends as a fuzzy ball, where each person may have a relationship with another, with varying degrees of strength. The same model would be used for sentences, each sentence would have a relationship with each other, with varying degrees of strength. 

And because I only had two weeks to do it, DJ'ing would probably be more feasible for a one-man team. 

# Strategy

If you've read the Lean Startup, you'll notice that Eric Ries advocates the "Minimum Viable Product" approach. In this context, my goal was to build a working model as fast as I could and then continuously iterate upon that. So the way I modeled this was:

1. Model one chapter
2. Model one book
3. Model five books
4. Model 10 books

And on... and on.... you get the idea.

# Rapid Automatic What.........? 

As I'm typing these words on the keyboard, I'm wondering how I can explain this without boring Machine-Learning enthusiasts while making it understandable for normal people.

Sorry ML people, general audience wins here.

Imagine yourself as a puzzle-maker. You're boss gives you a beautiful sunset photo and wants you to hand-cut the pieces out. Each time you cut out the pieces, you have a little snippet of that photo. In Natural Language Processing, taking a picture and cutting it into pieces is called tokenizing. In order to analyze text, we need to cut it up into different pieces( usually each piece = word) but it depends on the project you're working on. 

In the context of this project, I wanted to tokenize on key words. Sometimes, an author might use a phrase like "Moby Dick." "Moby Dick" should be treated as one phrase, not two. This is called Rapid Automatic Keyword Extraction. 

After passing my books & summaries through a Rapid Automatic Keyword Extraction, it's time to engineer features:

# Feature Engineering

To understand what's going on here, let me introduce you to this scenario:

Let's say, you're waiting for your Uber to come and have a couple minutes of time to kill. So you flip out your phone and open up your Facebook. You start scrolling through your newsfeed and see that Sally posted an article that says "Trump Suggests Bigger Role for U.S. in Syria’s Conflict." You live in San Francisco, so you have a passionate hate for Trump, so you click on the link. The article is kinda long and since you're limited on time, you scan the article, trying to decide if the entire thing is worth reading. You see that the article talks about "North Korea" and start thinking " Ohh.... this is interesting, I'll save this for later." When you saw the title of the news article, what keyword triggered you to click?  Trump. If you're interested in foreign policy, it might've been Syria . It changes for each person, but the idea is that there were specific key words in the text that gave you a good picture of what the article is about. And when you scan the article, you see North Korea, so Trump, Syria, North Korea , already give you an idea that this article is about some problems/tensions. 

This idea of a key word giving you some information about text is called a feature. Features are kinda like hints. It's saying "HEY MODEL! PAY ATTENTION TO THIS A LITTLE BIT MORE!" 

In addition to key words, here are all the things I thought the model should notice: 

1. Term-Frequency: If a keyword appeared more often, the better the sentence. 

2. Sentence Location: Sentences in the beginning are likely to be more important, since the author is often introducing the general concept of the entire book. Middle sections are usually diving into details, examples of an idea, which may not be the best sentences for summarizing. 

3. Presence of a Verb: I used a position tagger to score the number of verbs a sentence contained. I guessed that sentences which contained verbs, likely had a subject-object action in the sentence, which usually provided more information and to get rid of flowery, descriptive sentences( which aren't good for summaries). 

4. Sentence Length: I down weighted short sentences, since a short, 4 word sentence, that might contain a key word, isn't that important. 

# Modeling

Latent Dirichlet Allocation: I felt that LDA would be the best topic model for extracting key words because a book covers a wide swath of topics, and the model all the words over all the topics. Other methods such as clustering, NMF couldn't do this. 

Here's a wordcloud of the chosen topic model for Chaos Monkey by Antonio Garcia Martinez:

![topic word cloud](https://cloud.githubusercontent.com/assets/22338112/24921620/4a1cec5a-1ea0-11e7-8e19-8216cfed709b.png)

And a wordcloud of the entire book:

![book word cloud](https://cloud.githubusercontent.com/assets/22338112/24921633/56cd0e3a-1ea0-11e7-9444-eb9abd186bbd.png)

I first modeled the books that I read, this way, I could hand-select the best topic for the book. I settled on 10 topics and 50 key words by examining summary quality for different parameters. 

The problem I faced when Topic-Modeling, is that each book had a different optimal topic. So when I moved onto books I haven't read, I wouldn't know what the ideal topic was. So I just chose a topic at random, with the understanding that it might affect my score in the future. 

In the future, a good idea would be to topic model all the books as one document which might be a better way of finding the best topic.   
	
Doc2Vec: In addition to a more Bayesian approach to modeling the text, I wanted to test a model neural-networky-y based approach. Similar to word2vec, except I'll be turning each sentence into a "document" and extracting the vectors for analysis.

I used Doc2Vec by transforming each sentence into sentence vectors and then took the cosine similarity of the vectors against the entire book to find the best sentences.

# Scoring

Summary's are subjective. There is no "correct" answer. So the challenge was to figure out how to quantify a "good" summary when there was no real benchmark. 

Based on my research, the ROUGE-N score was the best measure for automatic summarizers ( compared to Latent Semantic Analysis).

When scoring the models, I wanted to test the models on the entire book and applying the model to 10 different sections of the book, and then aggregating the summary. Here are the scores:

Doc2Vec Split in 10: 0.241( + 0.126 over random)

LDA Split in 10: 0.176 ( + 0.062 over random)

Random Split in 10: 0.114 (-----------------)

# Conclusions

1. Doc2Vec Worked Best: As you can see by the scores, the Doc2Vec split in 10 model seemed to score the highest. Just by looking at the scores, splitting the book into 10 sections seemed to immediately boost the score, compared to applying the model to the entire book. 

2. Better to be used for previews than summaries: Because I chose an extraction-based method, I was already aware that the writing style of an author compared to a human summarizer was going to be a bit different. 

To dive a bit deeper into the writing style, author's write their books, knowing that they have approx 300 pages to convey their idea. As a result, sentences will contain much more detail, and author's are willing to dive into technicalities a bit more because they have enough space to explain a term they can use for the rest of the book. Compared with the human-written summaries, human writers are obviously going to condense the writing into fewer words, while diluting the arguments behind the concepts. 

3. Model is biased towards long sentences: If you look at the average sentence length:

![word per sentence](https://cloud.githubusercontent.com/assets/22338112/24921178/c60505e8-1e9e-11e7-8d0a-c0b7e1c71bd5.png)

In addition, I created a quick regression of word/sentence against ROUGE-N score. 

![word sentence regression](https://cloud.githubusercontent.com/assets/22338112/24921114/9626393c-1e9e-11e7-925f-8b79253deb8b.png)

You'll notice that the average words/sentence for the Doc2Vec summaries are about 20 words/sentence longer than the words/sentence in the reference summaries. This finding leads me to claim, that the model bias' a bit towards longer sentences, which makes sense due to the scoring method. A longer sentence has a higher likelihood of containing pairs of words that match pairs in the reference summaries, which boosts the ROUGE-N score. This method does eliminate low-information short sentences.

In the future, the problem will be to figure out how not to overweight the long sentences while still eliminating the short sentences.  

4. Human summarizers emphasize different key points: Summaries and most writing, is subjective. A human summarizer already decides upon what key points they think the reader finds interesting. However, every reader is asking different questions when they're reading a book. An older man, may be wondering how he can find peace for the rest of his life, while a teenage girl may be trying to figure out what she should do with her life. Different questions, different answers, different summaries. 

A future solution can be a query-based summarization method, where the user inputs a specific question they're asking, and then the model writes the summary based on the question the user asks. 

# Future

In the future, there are many things I may be able to try:

1. Learn summarization framework: Similar to the grade-school five paragraph format, I can teach the Booki Monster a summarization-writing framework. This can improve the coherence and flow of ideas within the summary.

2. Human Feedback: Scoring is hard. Like I said before, summaries are subjective. In the future, having the model create summaries and get user feedback can add a human element to summary creation.

3. Query-Based Summary: Have users input questions and model creates summary based on those questions

# Example

Visit www.bookimonster.com to see sample summaries of a few books. 

Web-Application coming soon!

# Sources

Automatic Extraction Based Summarizer - R.M Aliguliyev

Latent Dirichlet Allocation Based Multi-Document Summarization - Rachit Arora, Balaraman Ravindran

Looking for a Few Good Metrics: ROUGE and its Evaluation - Chin-Yew Lin

Sentence Extraction Based Single Document Summarization - Jagadeesh J, Prasad Pingali, Vasudeva Varma

Distributed Representations of Sentences and Documents - Quoc Le, Tomas Mikolov

Latent Dirichlet Allocation - David Blei, Andrew Ng, Michael Jordan

LDA2Vec - Chris Moody
