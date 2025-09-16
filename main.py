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

    # THIS DOESN"T SEEM TO BE WORKING.
    for word in tweet_split:
        # Very simple URL detection.  
        if word not in stopwords.words("english") or not (word[0] == "@" or word[0] == "#"): #and "http" in word and ".com" in word and ".net" in word
            tweet_split_processed.append(word)

    tweet = ""
    for word in tweet_split_processed:
        tweet = tweet + " " + str(word)

    # Remove punctuation
    #tweet_chars = [char for char in tweet if not (char in string.punctuation or char == "\n")]

    #tweet = ""
    #for char in tweet_chars:
    #    tweet = tweet + str(char)
    
    # Convert emojis (if applicable).

    # Return the processed tweet.
    print(tweet)
    return tweet

data["tweet"].apply(data_preprocessing)

print(data)

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