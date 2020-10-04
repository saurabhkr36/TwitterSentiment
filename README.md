# TwitterSentiment
## Objective:
The main objective of this project is to predict the tweets of test set developed from main dataset as positive (4) or negative (0). As per the dataset, It is assumed that any tweet with positive emoticons, like :), were positive, and tweets with negative emoticons, like :(, were negative. Based on this primary decisive feature the tweets are categorized as positive(supportive/happy) or negative(opposing/sad).

## Introduction:
It was built from the Sentiment140 dataset(https://www.kaggle.com/kazanova/sentiment140?select=training.1600000.processed.noemoticon.csv) dataset available on Kaggle, but this dataset offers a binary classification of the classified sentiment.

The link to the Sentiment140 dataset contains this information about the contents:

"Context:

This is the sentiment140 dataset. It contains 1,600,000 tweets extracted using the twitter api . The tweets have been annotated (0 = negative, 4 = positive) and they can be used to detect sentiment.

Content:

It contains the following 6 fields:

-target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)

-ids: The id of the tweet ( 2087)

-date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)

-flag: The query (lyx). If there is no query, then this value is NO_QUERY.

-user: the user that tweeted (robotickilldozr)

-text: the text of the tweet (Lyx is cool)"

## Tweets Preprocessing and Cleaning
From the main dataset, the 'ID', 'User_ID', 'Date' and 'query' columns are removed as they are of no use in this objective. Alot of punctuations and words without contextual meanings are in the tweets which needs to be removed. There are several other user mentions, hyperlink texts, emoticons, etc. which has no use as a feature for our model and needs to be removed.
->Stemming: It usually refers to a process that chops off the ends of words in the hope of achieving goal correctly most of the time and often includes the removal of derivational affixes.
->Lemmatization: It usually refers to doing things properly with the use of a vocabulary and morphological analysis of words, normally aiming to remove inflectional endings only and to return the base and dictionary form of a word.
The goal of both stemming and lemmatization is to reduce inflectional forms and sometimes derivationally related forms of a word to a common base form.
->Stopwords are commonly used words in English which have no contextual meaning in an sentence. So therefore, remove them before classification as well.
I've used NLTK library for text preprocessing.

## Tokenization and Word embedding
The preprocessed data is now tokenized to get individual words.
Given a character sequence and a defined document unit, tokenization is the task of chopping it up into pieces, called tokens , perhaps at the same time throwing away certain characters, such as punctuation. The process is called Tokenization.
A tokenizer object can be used to covert any word into a Key in dictionary (number).
Word Embedding is one of the popular representation of document vocabulary.It is capable of capturing context of a word in a document, semantic and syntactic similarity, relation with other words, etc. Basically, it's a feature vector representation of words which are used for other natural language processing applications.
Here, I've use Transfer Learning. The pretrained Word Embedding like GloVe & Word2Vec gives more insights for a word which can be used for classification. I've downloaded the GloVe embedding and used it in this model.

## LSTM Model
While building a deep learning model, three things needs to be emphasized on: Model Architecture, Hyperparmeter Tuning and Performance of the model. Since sequence model gives better results compared to ML algorithms like naive bayes, Single value decomposition, etc. Thus, I've used Long Short Term Memory (LSTM) model in this dataset.
In this model, the neural network outputs a scalar value prediction instead of a sequence value prediction.
->For model architecture, we use

1) Embedding Layer - Generates Embedding Vector for each input sequence.

2) Conv1D Layer - Its using to convolve data into smaller feature vectors.

3) LSTM - Long Short Term Memory, its a variant of RNN which has memory state cell to learn the context of words which are at further along the text to carry contextual meaning rather than just neighbouring words as in case of RNN.

4) Dense - Fully Connected Layers for classification

-> Optimization algorithm used: Adam optimization for gradient descent.

->Callbacks are special functions which are called at the end of an epoch. We can use any functions to perform specific operation after each epoch. I used two callbacks here,

1) LRScheduler - It changes a Learning Rate at specfic epoch to achieve more improved result. In this notebook, the learning rate exponentionally decreases after remaining same for first 10 Epoch.

2) ModelCheckPoint - It saves best model while training based on some metrics. Here, it saves the model with minimum Validity Loss.
A threshold of 0.5 is given to final output to predict 0 (if probability<0.5) and 4(if probability>=0.5).


## End Notes
In text preprocessing stage, stemming, lemmatization are used along with removal of stopwords.
The preprocessed texts are tokenized, embedded and feeded into LSTM network connected to Dense layers and sigmoid activation function to predict final probability output which is further have a threshold of 0.5 to predict 0 (if <0.5) and 4(if >=0.5).
