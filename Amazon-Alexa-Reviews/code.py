# --------------
# import packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

# Load the dataset
df = pd.read_csv(path, sep ="\t")
print(type(df['date']))

# Converting date attribute from string to datetime.date datatype 
df['date']= pd.to_datetime(df['date']) 


# calculate the total length of word
df['length'] = df['verified_reviews'].apply(lambda x: len(x))





# --------------
## Rating vs feedback
import matplotlib.pyplot as plt 
import seaborn as sns 
# set figure size
plt.figure(figsize=(15,7))


df.head()

# generate countplot
sns.countplot(x = 'rating', hue = 'feedback' , data = df)

# display plot
sns.barplot(x = 'rating',y = "variation", hue = 'feedback', data = df)


## Product rating vs feedback

# set figure size


# generate barplot


# display plot




# --------------
# import packages
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer

# declare empty list 'corpus'
corpus = []
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
# for loop to fill in corpus
for review in df['verified_reviews']:
    # retain alphabets
    # convert to lower case
    raw = review.lower()
    # tokenize
    tokens = tokenizer.tokenize(raw)
    # initialize stemmer object
    ps = PorterStemmer()
    # perform stemming
    stopped_tokens = [token for token in tokens if token not in stop_words]
    stemmed_tokens = [ps.stem(token) for token in stopped_tokens]
    # join elements of list
    corpus.append(" ".join(stemmed_tokens))
    # add to 'corpus'
    
    
# display 'corpus'
print(corpus)


# --------------
# import libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Instantiate count vectorizer
cv = CountVectorizer(max_features = 1500)

# Independent variable
X = cv.fit_transform(corpus).toarray()


# dependent variable
y = df['feedback']

# Counts
count = df['feedback'].value_counts()

# Split the dataset
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)




# --------------
# import packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score

# Instantiate calssifier
rf = RandomForestClassifier(random_state = 2)

# fit model on training data
rf.fit(X_train,y_train)

# predict on test data
y_pred = rf.predict(X_test)

# calculate the accuracy score
score = accuracy_score(y_test,y_pred)

# calculate the precision
precision = precision_score(y_test,y_pred)

# display 'score' and 'precision'
print(score)
print(precision)



# --------------
# import packages
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score

# # Instantiate smote
# smote = SMOTE(random_state = 9)

# # fit_sample onm training data
# X_train_s, y_train_s = smote.fit_sample(X_train,y_train)

# # fit modelk on training data
# rf.fit(X_train_s,y_train_s)

# # predict on test data
# y_pred = rf.predict(X_test)

# # calculate the accuracy score
# score = accuracy_score(y_test, y_pred)

# # calculate the precision
# precision = precision_score(y_test, y_pred)

# # display precision and score
# print(score)
# print(precision)

# Instantiate smote
smote = SMOTE(random_state=9)

# fit_sample onm training data
X_train, y_train = smote.fit_sample(X_train, y_train)

# fit modelk on training data
rf.fit(X_train, y_train)

# predict on test data
y_pred = rf.predict(X_test)

# calculate the accuracy score
score = accuracy_score(y_test, y_pred)

# calculate the precision
precision = precision_score(y_test, y_pred)

# display precision and score
print(score, precision)


