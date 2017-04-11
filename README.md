# The Booki Monster
![alt tag](http://www.babydigezt.com/wp-content/uploads/2014/09/book-monster.jpg)

One of the biggest problems I've noticed since entering the "real world", is that it's hard to find time for things. Going on adventures, finding a girlfriend, building new social groups, having interesting hobbies...... and reading books. Or maybe, you're just too hungover....... 

But a book that might take 5-10 hours to read? Well, what if I told you that you could get the key points from the book without actually reading the book? I want to introduce you to the "Booki Monster." A machine-learning powered monster that reads your books and summarizes them for you. 

While writing this up, I had two audiences in mind:

- The " I barely know what Machine Learning is" crowd.
- The "I am a Machine Learning genius" crowd. 

So my goal here is to make part of this post relatable to a general audience. Anytime I intend to dive into more technical language, I'll let you know so you're not completely confused :) 

# 1.Who is the Booki Monster?

You see those pile of books on your drawer that you're "too busy" to crack open? Well, that's the Booki Monster's food. Feed the Booki Monster your books and then she'll spit out the golden nuggets in the form of summaries.

# 2.Feeding the Monster

You know that feeling when you're with your friends, you want to eat out, but can't decide where to eat because there are wayyyy too many options? Feeding the Booki Monster was the same, I had too many options: science fiction, business, self-help, psychology, scientific research etc. 

And for those who understand product marketing, when you're product is for everyone, it's for no one.  I'd prefer to make the Booki Monster generate high-quality summaries for a more niche, targeted set of books, than mediocre summaries for many books. 

So I settled on feeding the Booki Monster only business books for this reason. Plus, Blinkist.com, a company that produces human-written summaries, happily agreed to send over their human-written summaries, so I can measure the Booki Monster summaries quality. 

With this understanding, grab your nearest surgeon, and let's start dissecting the body of the Booki Monster. Mmmmm.... tasty........

# 3.The Booki Monster's Body (Non-Technical)

To understand how the Booki Monster's body actually produces summaries, let's talk about how you read on the internet. 

Let's say, you're waiting for your Uber to come and have a couple minutes of time to kill. So you flip out your phone and open up your Facebook. You start scrolling through your newsfeed and see that Sally posted an article that says "Trump Suggests Bigger Role for U.S. in Syria’s Conflict." You live in San Francisco, so you have a passionate hate for Trump, so you click on the link. The article is kinda long and since you're limited on time, you scan the article, trying to decide if the entire thing is worth reading. You see that the article talks about "North Korea" and start thinking " Ohh.... this is interesting, I'll save this for later." When you saw the title of the news article, what keyword triggered you to click?  Trump. If you're interested in foreign policy, it might've been Syria . It changes for each person, but the idea is that there were specific key words in the text that gave you a good picture of what the article is about. And when you scan the article, you see North Korea, so Trump, Syria, North Korea , already give you an idea that this article is about some problems/tensions. 

In relation to the Booki Monster, the monster will find these important keywords in an article that'll give you the most amount of information. And the assumption here, is that sentences that have these keywords are likely to give you the highest amount of information about the article( which isn't always true). 

Similar to key words, I also looked at:

1.Sentence Location: Sentences that are topic sentences are probably going to give you more information.
2.Verb: Sentences that contain verbs are telling you an action occuring, so it's probably going to give you more information.
3.Sentence Length: If a sentence is four words, it probably won't tell you as much than a 10 word sentence.    

Using all these factors, I created a sentence score. Then, I ranked the sentences based on the score. Then the Booki Monster would spit out the sentences that had the highest scores. 

# 3.The Booki Monster's Body (Technical)

# Method

When creating the Booki Monster, I had a couple different options:

1. Sentence Extraction
2. Abstractive Methods
3. Graph-Based

I decided that within my timeframe of two weeks, an extraction-based methodology would be most feasible( especially for a one-person team). 

After selecting the methodology, I noticed in books, certain phrases of words tended to carry  more weight than just the words itself. So in addition to removing the stop words, I passed the data through a Rapid Automatic Keyword Extraction which tokenized my sentences by key phrases, not solely words. 

I took an iterative approach to modeling. I'd model a paragraph, then a chapter and then the entire book. 

After tokenizing my text files, I engineered four different features for the sentences:

# Feature Engineering

1. Term-Frequency: If a keyword appeared more often, the better the sentence. 

2. Sentence Location: Sentences in the beginning are likely to be more important, since the author is often introducing the general concept of the entire book. Middle sections are usually diving into details, examples of an idea, which may not be the best sentences for summarizing. 

3. Presence of a Verb: I used a position tagger to score the number of verbs a sentence contained. I guessed that sentences which contained verbs, likely had a subject-object action in the sentence, which usually provided more information and to get rid of flowery, descriptive sentences( which aren't good for summaries). 

4. Sentence Length: I down weighted short sentences, since a short, 4 word sentence, that might contain a key word, isn't that important. 

# Modeling

Latent Dirichlet Allocation: I felt that LDA would be the best topic model for extracting key words because a book covers a wide swath of topics, and the model all the words over all the topics. Other methods such as clustering, NMF couldn't do this. 

Here's a wordcloud of the topic models for Chaos Monkey by Antonio Garcia Martinez:

I first modeled the books that I read, this way, I could hand-select the best topic for the book. I settled on 10 topics and 50 key words by examining summary quality for different parameters. 
The problem I faced when Topic-Modeling, is that each book had a different optimal topic. So when I moved onto books I haven't read, I wouldn't know what the ideal topic was. So I just chose a topic at random, with the understanding that it might affect my score in the future. 

In the future, a good idea would be to topic model all the books as one document which might be a better way of finding the best topic.   
	
Doc2Vec: In addition to a more Bayesian approach to modeling the text, I wanted to test a model neural-networky-y based approach. Similar to word2vec, except I'll be turning each sentence into a "document" and extracting the vectors for analysis.

I used Doc2Vec by transforming each sentence into sentence vectors and then took the cosine similarity of the vectors against the entire book to find the best sentences.

# Scoring

Summary's are subjective. There is no "correct" answer. So the challenge was to figure out how to quantify a "good" summary when there was no real benchmark. 

Based on my research, the ROUGE-N score was the best measure for automatic summarizers ( compared to Latent Semantic Analysis).

When scoring the models, I wanted to test the models on the entire book and applying the model to 10 different sections of the book, and then aggregating the summary. Here are the scores:

![scores](https://cloud.githubusercontent.com/assets/22338112/24921319/4486c2e4-1e9f-11e7-8169-54a82efcd47f.png)

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
