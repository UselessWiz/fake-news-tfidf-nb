import numpy as np 
import pandas as pd 
import string

from scipy.sparse import hstack
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

"""
--------------------
Function Definitions
--------------------
"""

# Returns True if word contains a URL, or False if not.
def url_check(word: str):
    return "http" in word and (".com" in word or ".net" in word)

def data_preprocessing(tweet: str, remove_hashtags: bool) -> str:
    # Convert to lowercase
    tweet = str(tweet).lower()
    tweet_split = tweet.split(" ")
    tweet_split_processed = []

    for word in tweet_split: 
        # Remove stopwords, handles, hashtags and urls.
        if remove_hashtags:
            if word not in stopwords.words("english") and not (word[0] == "@" or word[0] == "#") and \
            not url_check(word): 
                tweet_split_processed.append(word)
        else:
            if word not in stopwords.words("english") and not word[0] == "@" and not url_check(word): 
                tweet_split_processed.append(word)

    tweet = ""
    for word in tweet_split_processed:
        tweet = tweet + " " + str(word)

    # Remove punctuation
    tweet.replace("\n", " ") # Remove new line characters
    tweet_chars = [char for char in tweet if not (char in string.punctuation)]

    tweet = ""
    for char in tweet_chars:
        tweet = tweet + str(char)

    # Return the processed tweet.
    return tweet

def avg_word_len(tweet: str):
    words = tweet.split()
    try:
        return sum(len(word) for word in words) / len(words)
    except ZeroDivisionError:
        return 0

def run_experiment() -> float:
    # Initialise the data.
    data = pd.read_csv("tweets.csv", encoding='latin-1') # tweets.csv has 2 columns, tweet and label

    # Preprocessing/Data cleaning
    data.dropna(axis=0, subset=["tweet"], inplace=True)
    data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4", "Unnamed: 5"], axis=1)

    # Feature Construction
    data["tweet"] = data["tweet"].apply(data_preprocessing, remove_hashtags=False)

    # Convert text to TF-IDF representation for text feature.
    vectorizer = TfidfVectorizer()
    tfidf_data = vectorizer.fit_transform(data["tweet"])

    # Collect all the features into a single table.
    all_data = hstack((tfidf_data, data[data.columns[2:]]))

    # Split training/testing.
    train_data, test_data, train_label, test_label = train_test_split(all_data, data["label"], train_size=0.8, random_state=67)

    # Find the best Naive Bayes alpha value and create a model with this from training data and labels.
    nb_alphas = {"alpha": list(np.arange(1/100000, 5, 0.0001))}
    fake_news_detection_model = GridSearchCV(MultinomialNB(), nb_alphas, cv=5, n_jobs=-1, verbose=1).fit(train_data, train_label)

    # Predict whether the test data is real or fake.
    prediction = fake_news_detection_model.predict(test_data)

    # Output statistics about the model's performance.
    model_confusion_matrix = confusion_matrix(test_label, prediction)
    model_confusion_matrix = pd.DataFrame(data=model_confusion_matrix, columns=["Predicted False", "Predicted True"], index=["Actual False", "Actual True"])
    model_accuracy = accuracy_score(prediction, test_label)
    print(f"Alpha: {fake_news_detection_model.best_estimator_.alpha}\t Model Accuracy: {model_accuracy}\nConfusion Matrix:\n{model_confusion_matrix}\n")
    
    # Return the model accuracy which can be used to judge which feature set performs best.
    return float(model_accuracy)

"""
-----------
Entry Point
-----------
"""

if __name__ == "__main__":
    run_experiment()