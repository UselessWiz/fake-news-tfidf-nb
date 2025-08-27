import numpy as np 
import pandas as pd 
import string

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

data = pd.read_csv("tweets.csv", encoding='latin-1') # tweets.csv has 2 columns, tweet and label

# Preprocessing/Data cleaning

data.dropna(subset=["tweet"], inplace=True)

def data_preprocessing(tweet: str) -> str:
    # Convert to lowercase
    tweet = str(tweet).lower()

    # Remove URLs, hashtags & handles & remove stopwords
    tweet_split = tweet.split(" ")
    tweet_split_processed = []

    for word in tweet_split:
        if not (word in stopwords.words("english") and word[0] == "@" and word[0] == "#" and "http" in word and ".com" in word and ".net" in word): # Very simple URL detection.
            tweet_split_processed.append(word)

    tweet = ""
    for word in tweet_split_processed: #work out a neater way to do this
        tweet = tweet + " " + str(word)

    # Remove punctuation
    tweet_chars = []
    for char in tweet:
        if not (char in string.punctuation and char =="\n"):
            tweet_chars.append(char)

    tweet = ""
    for char in tweet_chars: # work out a neater way to do this - list comprehension?
        tweet = tweet + str(char)

    return tweet

    # Convert emojis (if applicable).

data["tweet"].apply(data_preprocessing)

# Feature Construction - convert to TF-IDF representation

vectorizer = TfidfVectorizer()
tfidf_data = vectorizer.fit_transform(data["tweet"])

train_data, test_data, train_label, test_label = train_test_split(tfidf_data, data["label"], train_size=0.8, random_state=67)
print(data["label"].shape)

# Train and test with Naive Bayes

nb_alphas = {"alpha": list(np.arange(1/100000, 100, 0.11))}

fake_news_detection_model = GridSearchCV(MultinomialNB(), nb_alphas, cv=5, n_jobs=-1, verbose=1).fit(train_data, train_label)

prediction = fake_news_detection_model.predict(test_data)
model_accuracy = accuracy_score(prediction, test_label)
print(f"Alpha: {fake_news_detection_model.best_estimator_.alpha}\t Model Accuracy: {model_accuracy}")