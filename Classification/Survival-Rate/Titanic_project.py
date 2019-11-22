# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
#############################################################################
# import dataset
da_train = pd.read_csv('train.csv')
da_train.head()
da_train.columns
da_test = pd.read_csv('test.csv')
#############################################################################
# data preprocessing

# replace missing values
da_train['Age'] = da_train['Age'].replace(np.nan, da_train['Age'].mean())
da_test['Age'] = da_test['Age'].replace(np.nan, da_test['Age'].mean())
da_test['Fare'] = da_test['Fare'].replace(np.nan, da_test['Fare'].mean())

da_train['Embarked'] = da_train['Embarked'].replace(np.nan, 'S')
da_test['Embarked'] = da_test['Embarked'].replace(np.nan, 'S')
#############################################################################
# plot variables to get intuition on their relationships
plt.scatter(da_train.Fare, da.Survived)
sns.scatterplot(da_train.Fare, da.Survived)

plt.hist(x = da_train.Age, bins = 30)
sns.distplot(da_train.Age)

sns.countplot(x = 'Pclass', hue = 'Survived', data = da_train)
sns.countplot(x = 'Sex', hue = 'Survived', data = da_train)
sns.countplot(x = 'SibSp', hue = 'Survived', data = da_train)
sns.countplot(x = 'Parch', hue = 'Survived', data = da_train)
sns.countplot(x = 'Embarked', hue = 'Survived', data = da_train)
#############################################################################
# get training and test sets
from sklearn.model_selection import train_test_split
x_col_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
y_col_names = ['Survived']
x_train, x_test, y_train, y_test = train_test_split(da_train[x_col_names], da_train[y_col_names], test_size = 0.2)
#############################################################################
# convert dummies
x_train = pd.get_dummies(x_train, drop_first = True).values
y_train = y_train.values[:, 0]
x_test = pd.get_dummies(x_test, drop_first = True).values
y_test = y_test.values[:, 0]
x_pred = pd.get_dummies(da_test[x_col_names], drop_first = True).values
#############################################################################
# classification models

##### (Base model - RnadomForest) #####
# Base model
X1 = x_train
y1 = y_train
X1_test = x_test
y1_test = y_test

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 200,
                    criterion = 'entropy',
                    max_depth = 20,
                    max_features = 7,
                    max_leaf_nodes = 50,
                    n_jobs = -1,
                    random_state=1)
model.fit(X1, y1)
y1_pred = model.predict(X1_test)

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
cm = confusion_matrix(y1, model.predict(X1))
accuracy_0 = accuracy_score(y1_test, y1_pred)
f1_0 = f1_score(y1_test, y1_pred)

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold as skf
score_base = cross_val_score(estimator = model,
                                     X = X1, y = y1,
                                     cv = skf(n_splits = 20))
accuracy_base = score_base.mean()

pred_0 = model.predict(x_pred)
output_0 = pd.DataFrame({'PassengerId': da_test.PassengerId, 'Survived': pred_0})
output_0.to_csv('my_submission_base.csv', index=False)

##### (SVC) #####
from sklearn.svm import SVC
classifier_svc = SVC(kernel = 'rbf', gamma = 'auto')
classifier_svc.fit(x_train, y_train)
y_pred_svc = classifier_svc.predict(x_test)

cm_1 = confusion_matrix(y_test, y_pred_svc)
accuracy_1 = accuracy_score(y_test, y_pred_svc)
precision_1 = precision_score(y_test, y_pred_svc)
recall_1 = recall_score(y_test, y_pred_svc)
f1_1 = f1_score(y_test, y_pred_svc)

score_svc = cross_val_score(estimator = classifier_svc,
                                     X = x_train, y = y_train,
                                     cv = skf(n_splits = 20))
accuracy_svc = score_svc.mean()

pred_1 = classifier_svc.predict(x_pred)
output_1 = pd.DataFrame({'PassengerId': da_test.PassengerId, 'Survived': pred_1})
output_1.to_csv('my_submission_svc.csv', index=False)

##### (K_NN) #####
from sklearn.neighbors import KNeighborsClassifier as KNN
classifier_knn = KNN(n_neighbors = 3)
classifier_knn.fit(x_train, y_train)
y_pred_knn = classifier_knn.predict(x_test)

cm_2 = confusion_matrix(y_test, y_pred_svc)
accuracy_2 = accuracy_score(y_test, y_pred_svc)
precision_2 = precision_score(y_test, y_pred_svc)
recall_2 = recall_score(y_test, y_pred_svc)
f1_2 = f1_score(y_test, y_pred_svc)

score_knn = cross_val_score(estimator = classifier_knn,
                                     X = x_train, y = y_train,
                                     cv = skf(n_splits = 20))
accuracy_knn = score_knn.mean()

pred_2 = classifier_knn.predict(x_pred)
output_2 = pd.DataFrame({'PassengerId': da_test.PassengerId, 'Survived': pred_2})
output_2.to_csv('my_submission_knn.csv', index=False)

##### (Decision tree) #####
from sklearn.tree import DecisionTreeClassifier
classifier_tree = DecisionTreeClassifier(criterion = 'entropy')
classifier_tree.fit(x_train, y_train)
y_pred_tree = classifier_tree.predict(x_test)

cm_4 = confusion_matrix(y_test, y_pred_tree)
accuracy_4 = accuracy_score(y_test, y_pred_tree)
precision_4 = precision_score(y_test, y_pred_tree)
recall_4 = recall_score(y_test, y_pred_tree)
f1_4 = f1_score(y_test, y_pred_tree)

score_tree = cross_val_score(estimator = classifier_tree,
                                     X = x_train, y = y_train,
                                     cv = skf(n_splits = 20))
accuracy_tree = score_tree.mean()

pred_4 = classifier_tree.predict(x_pred)
output_4 = pd.DataFrame({'PassengerId': da_test.PassengerId, 'Survived': pred_4})
output_4.to_csv('my_submission_tree.csv', index=False)

#### (Bagging) ####
from sklearn.ensemble import BaggingClassifier
classifier_bagging = BaggingClassifier(n_estimators = 10)
classifier_bagging.fit(x_train, y_train)
y_pred_bagging = classifier_bagging.predict(x_test)

cm_5 = confusion_matrix(y_test, y_pred_bagging)
accuracy_5 = accuracy_score(y_test, y_pred_bagging)
precision_5 = precision_score(y_test, y_pred_bagging)
recall_5 = recall_score(y_test, y_pred_bagging)
f1_5 = f1_score(y_test, y_pred_bagging)

score_bagging = cross_val_score(estimator = classifier_bagging,
                                     X = x_train, y = y_train,
                                     cv = skf(n_splits = 20))
accuracy_bagging = score_bagging.mean()

pred_5 = classifier_bagging.predict(x_pred)
output_5 = pd.DataFrame({'PassengerId': da_test.PassengerId, 'Survived': pred_5})
output_5.to_csv('my_submission_bagging.csv', index=False)

#### (Adaboost) ####
from sklearn.ensemble import AdaBoostClassifier as ABC
classifier_adaboost = ABC(base_estimator = classifier_tree)
classifier_adaboost.fit(x_train, y_train)
y_pred_adaboost = classifier_adaboost.predict(x_test)

cm_6 = confusion_matrix(y_test, y_pred_adaboost)
accuracy_6 = accuracy_score(y_test, y_pred_adaboost)
precision_6 = precision_score(y_test, y_pred_adaboost)
recall_6 = recall_score(y_test, y_pred_adaboost)
f1_6 = f1_score(y_test, y_pred_adaboost)

score_adaboost = cross_val_score(estimator = classifier_adaboost,
                                     X = x_train, y = y_train,
                                     cv = skf(n_splits = 20))
accuracy_adaboost = score_adaboost.mean()

pred_6 = classifier_adaboost.predict(x_pred)
output_6 = pd.DataFrame({'PassengerId': da_test.PassengerId, 'Survived': pred_6})
output_6.to_csv('my_submission_adaboost.csv', index=False)

# (GradientBoosting)
from sklearn.ensemble import GradientBoostingClassifier as GBC
classifier_GBC = GBC()
classifier_GBC.fit(x_train, y_train)
y_pred_GBC = classifier_GBC.predict(x_test)

cm_7 = confusion_matrix(y_test, y_pred_GBC)
accuracy_7 = accuracy_score(y_test, y_pred_GBC)
precision_7 = precision_score(y_test, y_pred_GBC)
recall_7 = recall_score(y_test, y_pred_GBC)
f1_7 = f1_score(y_test, y_pred_GBC)

score_GBC = cross_val_score(estimator = classifier_GBC,
                                     X = x_train, y = y_train,
                                     cv = skf(n_splits = 20))
accuracy_GBC = score_GBC.mean()

pred_7 = classifier_GBC.predict(x_pred)
output_7 = pd.DataFrame({'PassengerId': da_test.PassengerId, 'Survived': pred_7})
output_7.to_csv('my_submission_gbc.csv', index=False)

#### (xgboost) ####
from xgboost import XGBClassifier
classifier_xgbc = XGBClassifier(booster = 'gbtree', learning_rate = 0.5, n_estimators = 100)
classifier_xgbc.fit(x_train, y_train)
y_pred_XGBC = classifier_xgbc.predict(x_test)

cm_8 = confusion_matrix(y_test, y_pred_XGBC)
accuracy_8 = accuracy_score(y_test, y_pred_XGBC)
precision_8 = precision_score(y_test, y_pred_XGBC)
recall_8 = recall_score(y_test, y_pred_XGBC)
f1_8 = f1_score(y_test, y_pred_XGBC)

score_XGBC = cross_val_score(estimator = classifier_xgbc,
                                     X = x_train, y = y_train,
                                     cv = skf(n_splits = 20))
accuracy_XGBC = score_XGBC.mean()

pred_8 = classifier_xgbc.predict(x_pred)
output_8 = pd.DataFrame({'PassengerId': da_test.PassengerId, 'Survived': pred_8})
output_8.to_csv('my_submission_xgb.csv', index=False)
#############################################################################
# model/parameters optimization for (base model)

# grid search
from sklearn.model_selection import GridSearchCV

parameters_base = [{'n_estimators': [150, 200, 300],
                    'max_depth': [20, 25, 40],
                    'max_features': (6, 7, 8),
                    'max_leaf_nodes': [50]
                    }]

grid_search_base = GridSearchCV(iid = False,
                                estimator = model,
                                param_grid = parameters_base,
                                scoring = 'accuracy',
                                n_jobs = -1,
                                cv = 10)
grid_search_base.fit(x_train, y_train)

grid_search_base.best_score_
grid_search_base.best_params_