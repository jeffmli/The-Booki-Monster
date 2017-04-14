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

And because I'm DJ'ing(extracting), the goal is for each sentence to get it's own "score." It's like how Steph Curry averages 25.3 points, 6.6 assists, 4.5 rebounds per game. Each one of these stats is a "feature" for Steph. And ESPN uses these numbers(plus many more) to create a PER score, for Steph it's 24.74. I'm trying to create the PER of sentences. 

And as you might be able to guess, there are an INFINITE amount of additional things I can track, here are a couple:

Sentence Structure: How many subjects, objects occur in the sentence? What combination of subject-object, verb, adverbs are most conducive to high-quality summary sentences?

Named Entity Tagging: If I'm reading an article about "San Francisco" and I see the word "San Francisco","Oakland","San Jose", should I give more weight to these special "entitites"? 

Sentence position within paragraph: Topic sentences should be upweighted while the middle sentences should be down-weighted.

PageRank: Similar to how Google's Search algorithm worked, I could add a PageRank method to additionally weight scores.

Word Length: Do # of characters in a word play a part in high-quality summaries?

Punctuation: How effective are rhetorical questions, questions, normal statements, exclamations in providing high-quality information to summaries?

And if the list keeps going, I'm either boring you, or I'm just trying to show you how smart I am(which if you're an ML engineer, you probably don't even think my modeling was smart. Well you're wrong.).

Anyways, let's get to the sexy stuff in Data Science. We've got our data, we've got our "features", what do we do next.... drumroll please............

# Modeling

You remember you're first few years in college, you're excited to become independent from your parents, so you get to your dorm room, your floormates become your bestest buddies, while you go on to inebriate yourself while riding the wave of independent life? And five years later, the wave crashes and you think back: " Man, I was an idiot. I would've totally made better use of my school resources, spent more time learning skills and put myself out there a bit more." 

I actually don't believe in replays because you wouldn't have known to do this, if you hadn't done that, circular logic. And the same goes for modeling book summaries. First time around, I'm super excited to add to the hype fueling the buzzword "Machine Learning." But as a young Data Scientist, I am young Luke Skywalker and have many, many things to learn. 

I'm going to show you what I did, and what I would do differently next time. 

But to dive in, I used two different models:

1. Latent Dirichlet Allocation

2. Doc2Vec

And you'll probably have no idea what those mean. Let's start with what Latent Dirichlect Allocation is and why I used it:

Let's say we took the world's 8 billion people and threw em all in a pot. Mixed them up all together. All the Asians standing next to each other, the Indians mixed with Arabs, English & Americans mixed, confusing as shit right. And let's say you were some almighty god, and Zeus commanded you to re-organize this pot into all their original countries without going one by one. How the eff would you do that?

Well, you would use a Topic Model. If you imagine each word in a text as a person, a word likely corresponds to a specific topic. For example, in an article about food, the words "dumpling","fried rice","herbal tea","small eyes" would fall under one topic and "fat","burgers","french fries","obesity" might fall under another. Can you guess what topics they are? Yes, Chinese and American. 

I chose Latent Dirichlet Allocation, because it does this categorizing for me. 

And since this was my first Data Science project, I wanted to make sure I had a model up and running first and ran out of time in trying other topic models. Other ones I considered:

- Non-Negative Matrix Factorization

- Principal Component Analysis

- Singular Value Decomposition

And I'm not going to exhaust you by explaining what each of those are. But I chose LDA, because each word isn't bound to a specific topic, but each word gets a distribution over all the topics( Sorry, for the non-tech folks, I don't have a good explanation for that, yet). 

This model would give me the best key words I could use for my scoring(explained earlier).

Here's what it looks like visually:

Here's a wordcloud of the chosen topic model for Chaos Monkey by Antonio Garcia Martinez:

![topic word cloud](https://cloud.githubusercontent.com/assets/22338112/24921620/4a1cec5a-1ea0-11e7-8e19-8216cfed709b.png)

And a wordcloud of the entire book:

![book word cloud](https://cloud.githubusercontent.com/assets/22338112/24921633/56cd0e3a-1ea0-11e7-9444-eb9abd186bbd.png)

And you might be wondering, how the heck does the model know how many categories to give the text? How does it know how many key words it chooses? It doesn't. I have to decide that and this depends on my knowledge of the text I'm modeling. I optimized my model for 10 topics & 50 key words. And I chose the topic based on my knowledge of the book( if I read it) or I chose them at random. 

(Eff... getting tired writing..... time for a coffee break!) 

The second model I tried is a Doc2Vec, which, yes, don't get too excited, is a "neural network." *GASP* *GASP* *GASP*

I'm being silly. You know, I need to have fun writing this.

Ok. Imagine you're standing on the surface of earth, you've been single for way too long, and want to find your significant other by pulling a Goku and shining a Kame-Kame-Ha lightbeam towards the sky. You'll determine that your new girlfriend/boyfriend will shine their own light to the sky. The one that's most similar to your light, is the winner.

Sorry, that's the best explanation I can do right now and the metaphor does not fully represent Doc2Vec correctly. However, the idea is that every sentence is like a beam of light shining to the sky(vector) and we want to see how similar these vectors are to the vector of the entire book. This gives us the score. 

And this is how I modeled. In the future, I would:

1. Try a basic Logistic Regression: Can I classify a specific sentence as representative of a reference summary sentence? 

2. Try all the topic-modeling models listed: A wider variety of models and give me different insights on the text.

3. Acquire more data to turn it into a Convolutional Neural Network.

4. Try a sole PageRank/Graph-Based Model.

5. Use all the models as weights for a "final score" for each sentence based on different techniques. 

# Scoring

Who is a better athlete, Kobe Bryant or Tom Brady? Who is the better writer, Tolstoy or Hemmingway? Who is the better visionary, Steve Jobs or Bill Gates? What's better, Apple or Android? Better ad platform, Facebook or Google?

When you ask different people, you get different answers. And summaries are the same way. Is there a quantitatively sound way of saying "Yes. This summary is dope." 

No, there isn't. But we can try. After doing some research, I found that researchers use something called ROUGE-N Score to measure quality of summaries. 

But what the heck does this score actually measure? It looks at the pairs of words in my booki-monster summary and then checks how many times these words occur in the human-written summary. And then takes a ratio. 

Here are the scores:

Doc2Vec Split in 10: 0.241( + 0.126 over random)

LDA Split in 10: 0.176 ( + 0.062 over random)

Random Split in 10: 0.114 (-----------------)

Note: Random means, I built a model that randomly takes sentences from the book and titles it "the summary." Because what the heck is the point, if a monkey can write summaries just as good as the Booki Monster's.... and..... these numbers don't mean anything unless we have a baseline.  

As you can see, the Kame-Kame-Ha Method(Doc2Vec) did 12.6% better than random and LDA did 6% better than random. 

# Conclusions

1. Better to be used for previews than summaries: Because I was DJ'ing/extracting, I knew that writing style of an author is going to be different from a summary. Author's tend to write their books, knowing they have many, many pages to articulate an iea. As a result, sentences will contain more detail, and author's are willing to dive into technicalities a bit more because they have enough space to explain a term they can use for the rest of the book. Compared with the human-written summaries, human writers are  going to condense the writing into fewer words, while diluting the arguments behind the concepts. 

2. Booki Monster loves long, meaty sentences: If you look at the average sentence length:

![word per sentence](https://cloud.githubusercontent.com/assets/22338112/24921178/c60505e8-1e9e-11e7-8d0a-c0b7e1c71bd5.png)

In addition, I created a quick regression of word/sentence against ROUGE-N score to look at the relationship. 

![word sentence regression](https://cloud.githubusercontent.com/assets/22338112/24921114/9626393c-1e9e-11e7-925f-8b79253deb8b.png)

Notice that the average words/sentence for the Doc2Vec summaries are about 20 words/sentence longer than the words/sentence in the reference summaries, which supports my first point as well. This finding leads me to claim, that the model bias' a bit towards longer sentences, which makes sense due to the scoring method. A longer sentence has a higher likelihood of containing pairs of words that match pairs in the reference summaries, which boosts the ROUGE-N score. This method does eliminate low-information short sentences.

In the future, how will I not over-weight long sentences but still keep short sentences? 

4. Human summarizers emphasize different key points: Summaries and most writing, is subjective. A human summarizer already decides upon what key points they think the reader finds interesting. However, every reader is asking different questions when they're reading a book. An older man, may be wondering how he can find peace for the rest of his life, while a teenage girl may be trying to figure out what she should do with her life. Different questions, different answers, different summaries. 

A future solution can be a query-based summarization method, where the user inputs a specific question they're asking, and then the model writes the summary based on the question the user asks. 

# Future

In the future, there are many things I may be able to try:

1. Learn summarization framework: Similar to the grade-school five paragraph format, I can teach the Booki Monster a summarization-writing framework. This can improve the coherence and flow of ideas within the summary.

2. Human Feedback: Scoring is hard. Like I said before, summaries are subjective. In the future, having the model create summaries and get user feedback can add a human element to summary creation.

3. Query-Based Summary: Have users input questions and model creates summary based on those questions. 

All in all, I hope you enjoyed reading this as much as I had writing/building this project. My journey into the world of Data Science is only beginning and I'll be creating many more monsters to come!

# Example

Collapse by Jared Diamond

As for the complications, of course it’s not true that all societies are doomed to collapse because of environmental damage in the past, some societies did while others didn’t; the real question is why only some societies proved fragile, and what distinguished those that collapsed from those that didn’t. Some societies that I shall discuss, such as the Icelanders and Tikopians, succeeded in solving extremely difficult environmental problems, have thereby been able to persist for a long time, and are still going strong today.

Some of my Montana friends now say in retrospect, when we compare the multi-billion dollar mine cleanup costs borne by us taxpayers with Montana’s own meager past earnings from its mines, most of whose profits went to shareholders in the eastern U.S. or in Europe, we realize that Montana would have been better off in the long run if it had never mined copper at all but had just imported it from Chile, leaving the resulting problems to the Chileans! After living for so many years elsewhere, I found that it took me several visits to Montana to get used to the panorama of the sky above, the mountain ring around, and the valley floor below to appreciate that I really could enjoy that panorama as a daily setting for part of my life and to discover that I could open myself up to it, pull myself away from it, and still know that I could return to it.

One person said that Balaguer might have been influenced by exposure to environmentalists during early years in his life that he spent in Europe; one noted that Balaguer was consistently anti Haitian, and that he may have sought to improve the Dominican Republic’s landscape in order to contrast it with Haiti’s devastation; another thought that he had been influenced by his sisters, to whom he was close, and who were said to have been horrified by the deforestation and river siltation that they saw resulting from the Trujillo years; and still another person commented that Balaguer was already 60 years old when he ascended to the post-Trujillo presidency and 90 years old when he stepped down from it, so that he might have been motivated by the changes that he saw around him in his country during his long life.

Web-Application coming soon!

# Sources

Automatic Extraction Based Summarizer - R.M Aliguliyev

Latent Dirichlet Allocation Based Multi-Document Summarization - Rachit Arora, Balaraman Ravindran

Looking for a Few Good Metrics: ROUGE and its Evaluation - Chin-Yew Lin

Sentence Extraction Based Single Document Summarization - Jagadeesh J, Prasad Pingali, Vasudeva Varma

Distributed Representations of Sentences and Documents - Quoc Le, Tomas Mikolov

Latent Dirichlet Allocation - David Blei, Andrew Ng, Michael Jordan

LDA2Vec - Chris Moody
