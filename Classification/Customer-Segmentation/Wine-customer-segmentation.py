import pandas as pd
import numpy as np
import keras

da = pd.read_csv("Wine.csv")
# da.columns
x_col_names = ['Alcohol', 'Malic_Acid', 'Ash', 'Ash_Alcanity', 'Magnesium',
               'Total_Phenols', 'Flavanoids', 'Nonflavanoid_Phenols',
               'Proanthocyanins', 'Color_Intensity', 'Hue', 'OD280', 'Proline']
x = da.loc[:, x_col_names].values
y = da.iloc[:, -1].values

#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#from sklearn.compose import ColumnTransformer
#x1_encoder = LabelEncoder()
#x[:, 1] = x1_encoder.fit_transform(x[:, 1])
#x2_encoder = LabelEncoder()
#x[:, 2] = x2_encoder.fit_transform(x[:, 2])
#dummy = ColumnTransformer(transformers = [('encoder', OneHotEncoder(categories = 'auto'), [1])], remainder = 'passthrough')
##OneHotEncoder(categories = "auto", categorical_features = [1])
#x = dummy.fit_transform(x)
#x = x[:, 1:]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# PCA dimensionality reduction
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
exp_var = pca.explained_variance_ratio_

# LDA dimensionality reduction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
x_train = lda.fit_transform(x_train, y_train)
x_test = lda.transform(x_test)

# kernel PCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2, kernel = 'rbf')
kpca.fit_transform(x_train)
x_test = kpca.transform(x_test)

# Multiple classification methods
# logistic regression
from sklearn.linear_model import LogisticRegression
classifier_log = LogisticRegression(solver = 'lbfgs', random_state = 10)
classifier_log.fit(x_train, y_train)

# K-NN classification
from sklearn.neighbors import KNeighborsClassifier
classifier_neighbor = KNeighborsClassifier(n_neighbors = 5)
classifier_neighbor.fit(x_train, y_train)

# SVM classification
from sklearn.svm import SVC
classifier_svm = SVC(gamma = 'auto', random_state = 10)
classifier_svm.fit(x_train, y_train)

# Naive Bayes classification
from sklearn.naive_bayes import GaussianNB
classifier_NB = GaussianNB()
classifier_NB.fit(x_train, y_train)

# XGBoost classification
from xgboost import XGBClassifier
classifier_boost = XGBClassifier(random_state = 10)
classifier_boost.fit(x_train, y_train)

# ANN model
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
classifier_ANN = Sequential()
classifier_ANN.add(Dense(units = 3, input_dim = 2, activation = 'relu', kernel_initializer = 'glorot_uniform'))
#classifier_ANN.add(Dense(units = 3, activation = 'relu', kernel_initializer = 'glorot_uniform'))
classifier_ANN.add(Dense(units = 3, activation = 'softmax', kernel_initializer = 'glorot_uniform'))
classifier_ANN.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier_ANN.fit(x_train, to_categorical(y_train)[:, 1:], batch_size = 10, epochs = 25)

# result prediction
y_pred_log = classifier_log.predict(x_test)
y_pred_neighbor = classifier_neighbor.predict(x_test)
y_pred_boost = classifier_boost.predict(x_test)
y_pred_svm = classifier_svm.predict(x_test)
y_pred_NB = classifier_NB.predict(x_test)
y_pred_ANN = classifier_ANN.predict(x_test)
y_pred_ANN = (y_pred_ANN > 0.5)

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm_log = confusion_matrix(y_test, y_pred_log)
cm_neighbor = confusion_matrix(y_test, y_pred_neighbor)
cm_boost = confusion_matrix(y_test, y_pred_boost)
cm_svm = confusion_matrix(y_test, y_pred_svm)
cm_NB = confusion_matrix(y_test, y_pred_NB)
cm_ANN = confusion_matrix(to_categorical(y_test)[:, 1:], y_pred_ANN)

# k-fold cross validation
from sklearn.model_selection import cross_val_score

logscore = cross_val_score(scoring = 'accuracy', estimator = classifier_log, X = x_train, y = y_train, cv = 10)
accuracy_log = logscore.mean()

neighborscore = cross_val_score(scoring = 'accuracy', estimator = classifier_neighbor, X = x_train, y = y_train, cv = 10)
accuracy_neighbor = neighborscore.mean()

svmscore = cross_val_score(scoring = 'accuracy', estimator = classifier_svm, X = x_train, y = y_train, cv = 10)
accuracy_svm = svmscore.mean()

NBscore = cross_val_score(scoring = 'accuracy', estimator = classifier_NB, X = x_train, y = y_train, cv = 10)
accuracy_NB = NBscore.mean()

boostscore = cross_val_score(scoring = 'accuracy', estimator = classifier_boost, X = x_train, y = y_train, cv = 10)
accuracy_boost = boostscore.mean()

#ANNscore = cross_val_score(scoring = 'accuracy', estimator = classifier_ANN, X = x_train, y = y_train, cv = 10)
#accuracy_ANN = ANNscore.mean()
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy_ANN = accuracy_score(y_test, y_pred_ANN)
precision_ANN = precision_score(y_test, y_pred_ANN)
recall_ANN = recall_score(y_test, y_pred_ANN)
f1_ANN = f1_score(y_test, y_pred_ANN)
