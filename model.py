# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
  
import pandas as pd
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import nltk
nltk.download("stopwords")



with open("C:\\Users\\Prathamesh\\Desktop\\Side projects\\Spam classifier\\SMSSpamCollection.txt") as f:
    lines = f.readlines()    

messages = pd.DataFrame([i.split("\t") for i in lines], columns=["label", "message"])



# Stopwords are words in the messages like [the if in a to] which we have to remove
from nltk.corpus import stopwords

# For stemming purpose to find the base root of the word
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


from nltk.stem import WordNetLemmatizer 
  
lemmatizer = WordNetLemmatizer()



corpus = []

for sms in tqdm(messages["message"]):
    # Removing unnecessary punctuations, numbers and replacing them with space
    review = re.sub("[^a-zA-Z]", " ", sms)
    
    # Convert the message to lowercase
    review = review.lower()
    
    # Split each word and create a list
    review = review.split()
    
    # Removing all stopwords in english language from the message and getting to the root of each word
    review = [ps.stem(w) for w in review if w not in stopwords.words("english")]
    
    # Convert back to full sentence format after cleaning
    review = " ".join(review)
    
    # Append to corpus
    corpus.append(review)
    
    
    
    
    
    
    
    
    
    
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report




# Creating features using Tf-idf Vectorization
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(corpus).toarray()
y = pd.get_dummies(messages["label"]).iloc[:,1].values


# Splitting train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Compensate for minority classes
X_train_res, y_train_res = SMOTE().fit_resample(X_train, y_train)

# Fit model on resampled training data
model = MultinomialNB().fit(X_train_res, y_train_res)

# Generate predictions
y_pred_probabilities = model.predict_proba(X_test)
y_pred = model.predict(X_test)


print("Model accuracy : {} %".format(round(accuracy_score(y_test, y_pred)*100, 2)))



import joblib

pickle.dump(model, open("model.pkl", 'wb'))
pickle.dump(tfidf, open("cv.pkl", 'wb'))