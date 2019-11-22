import numpy as np
import pandas as pd

da = pd.read_csv("Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3)

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(len(da)):
    review = re.sub('[^a-zA-Z]', ' ', da['Review'][i])
    review = review.lower()
    review = review.split()
    PS = PorterStemmer()
    review = [PS.stem(w) for w in review if not w in stopwords.words("english")]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(corpus).toarray()
y = da.iloc[:, 1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10)
classifier.fit(x_train, y_train)
y_pred_forest = classifier.predict(x_test)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)
y_pred_tree = classifier.predict(x_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)
y_pred_naive = classifier.predict(x_test)

from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score, 
roc_auc_score, precision_score, recall_score, average_precision_score)
#cm = confusion_matrix(y_test, y_pred)
#roc_auc = roc_auc_score(y_test, y_pred)
#av_precision = average_precision_score(y_test, y_pred)
accuracy_forest = accuracy_score(y_test, y_pred_forest)
precision_forest = precision_score(y_test, y_pred_forest)
recall_forest = recall_score(y_test, y_pred_forest)
f1_forest = f1_score(y_test, y_pred_forest)

accuracy_tree = accuracy_score(y_test, y_pred_tree)
precision_tree = precision_score(y_test, y_pred_tree)
recall_tree = recall_score(y_test, y_pred_tree)
f1_tree = f1_score(y_test, y_pred_tree)

accuracy_naive = accuracy_score(y_test, y_pred_naive)
precision_naive = precision_score(y_test, y_pred_naive)
recall_naive = recall_score(y_test, y_pred_naive)
f1_naive = f1_score(y_test, y_pred_naive)