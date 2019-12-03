import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

da = pd.read_csv('creditcard.csv')
da.head()
da.info()
da.describe()
da.columns

x = da.iloc[:, :-1]
y = da.iloc[:, -1]

class_counts = y.value_counts()
print("Fraud rate is {}% of Total Transactions.".format(round(class_counts[1]/len(y)*100, 2)))
#############################################################################
plt.figure(figsize=(15, 15))
for i in range(x.shape[1]):
    plt.subplot(6, 5, i+1)
    sns.distplot(x[x.columns[i]])
    plt.title(x.columns[i])

corr = da.corr()
plt.figure(figsize=(20, 15))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
#############################################################################
# Train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size = 0.3, 
                                                    random_state = 5)
#############################################################################
def classification(X_Train, Y_Train, X_Test, Y_Test, cl_wt={0:1, 1:1}):
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.ensemble import GradientBoostingClassifier as GBC
    from xgboost import XGBClassifier
    from sklearn.ensemble import AdaBoostClassifier as ABC
    '''
    list of calssification models:
        'log': LogisticRegression(class_weight = cl_wt, solver = 'lbfgs',
                                  random_state = 5, n_jobs = -1)
        'knn': KNeighborsClassifier(n_neighbors = 3)
        'svc': SVC(class_weight = cl_wt, random_state = 5)
        'naive': GaussianNB()
        'decision_tree': DecisionTreeClassifier(class_weight = cl_wt, 
                                                criterion = 'entropy')
        'random_forest': RandomForestClassifier(class_weight = cl_wt,
                                                n_estimators = 200,
                                                criterion = 'entropy',
                                                max_depth = 20,
                                                max_leaf_nodes = 50,
                                                n_jobs = -1,
                                                random_state=1)
        'bagging': BaggingClassifier(n_estimators = 10)
        'gradient_boost': GBC()
        'xgb': XGBClassifier(random_state = 5)
        'adaboost': ABC(base_estimator = classifier_tree)
    '''
    models = {
                'log': LogisticRegression(class_weight = cl_wt, solver = 'lbfgs',
                                          random_state = 5, n_jobs = -1),
                'knn': KNeighborsClassifier(n_neighbors = 3),
                'svc': SVC(class_weight = cl_wt, random_state = 5),
                'naive': GaussianNB(),
                'decision_tree': DecisionTreeClassifier(class_weight = cl_wt, 
                                                criterion = 'entropy'),
                'random_forest': RandomForestClassifier(class_weight = cl_wt,
                                                n_estimators = 200,
                                                criterion = 'entropy',
                                                max_depth = 20,
                                                max_leaf_nodes = 50,
                                                n_jobs = -1,
                                                random_state=1),
                'bagging': BaggingClassifier(n_estimators = 10)                
              }
    
    accuracy = {}
    precision = {}
    recall = {}
    f1 = {}
    metrics = pd.DataFrame()
    for key, value in models.items():
        print(key, value)
        value.fit(X_Train, Y_Train)
        y_pred = value.predict(X_Test)
        accuracy[key] = accuracy_score(Y_Test, y_pred)
        precision[key] = precision_score(Y_Test, y_pred)
        recall[key] = recall_score(Y_Test, y_pred)
        f1[key] = f1_score(Y_Test, y_pred)
        new_metric = pd.DataFrame({'model': [key],
                                   'accuracy': [accuracy[key]],
                                   'precision': [precision[key]],
                                   'recall': [recall[key]],
                                   'f1': [f1[key]]})
        metrics = pd.concat([metrics, new_metric])
    return metrics
#############################################################################
# modeling - original data
original_metrics = classification(x_train, y_train, x_test, y_test)

#############################################################################
# Apply class-weight parameter
weighted_metrics = classification(x_train, y_train, x_test, y_test, 'balanced')

#############################################################################
# Try oversampling method
da_train = pd.concat([x_train, y_train], axis=1)

# separate minority and majority classes
legit = da_train[da_train.Class==0]
fraud = da_train[da_train.Class==1]

# oversample minority
from sklearn.utils import resample
fraud_oversampled = resample(fraud,
                          replace=True, # sample with replacement
                          n_samples=len(legit), # match number in majority class
                          random_state=1)

# combine majority and upsampled minority
oversampled_data = pd.concat([legit, fraud_oversampled])

x_train_oversample = oversampled_data.drop('Class', axis=1)
y_train_oversample = oversampled_data['Class']

# Build models with oversampled data
oversampled_metrics = classification(x_train_oversample, y_train_oversample, x_test, y_test)

#############################################################################
# Try undersampling method
legit_undersampled = resample(legit,
                                replace = False, # sample without replacement
                                n_samples = len(fraud), # match minority n
                                random_state = 5)

# combine minority and undersampled majority
undersampled_data = pd.concat([legit_undersampled, fraud])

x_train_undersample = undersampled_data.drop('Class', axis=1)
y_train_undersample = undersampled_data['Class']

# Build models with undersampled data
undersampled_metrics = classification(x_train_undersample, y_train_undersample, x_test, y_test)

#############################################################################
# SMOTE method (oversampling)
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=5, ratio='minority')

x_sm, y_sm = sm.fit_sample(x_train, y_train)

# Build models with oversampled data
smote_metrics = classification(x_sm, y_sm, x_test, y_test)

#############################################################################
# TomekLinks method (undersampling)
from imblearn.under_sampling import TomekLinks
tl = TomekLinks(return_indices=True, ratio='majority')
x_tl, y_tl, id_tl = tl.fit_sample(x_train, y_train)

# Build models with undersampled data
tomek_metrics = classification(x_tl, y_tl, x_test, y_test)

#############################################################################
# comparing different models
Metrics = pd.concat([original_metrics, weighted_metrics,
                     oversampled_metrics, undersampled_metrics,
                     smote_metrics, tomek_metrics])

Metrics['run_type'] = ['original']*4 + ['weighted']*4 + ['oversampled']*4 + ['undersampled']*4 + ['smote']*4 + ['tomek']*4
Metrics.set_index('run_type').sort_values(by='f1', ascending=False)
#############################################################################
# Building randomforest model
from sklearn.ensemble import RandomForestClassifier
final_model = RandomForestClassifier(class_weight = {0:1, 1:1},
                                     n_estimators = 200,
                                     criterion = 'entropy',
                                     max_depth = 20,
                                     max_leaf_nodes = 50,
                                     n_jobs = -1,
                                     random_state=1)
final_model.fit(x_train, y_train)
y_pred = final_model.predict(x_test)

# k-fold cross validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold as skf
score = cross_val_score(estimator = final_model,
                            scoring = 'f1',
                            X = x_train, y = y_train,
                            cv = skf(n_splits = 10))
f1 = score.mean()
#############################################################################
# grid search
from sklearn.model_selection import GridSearchCV

parameters = [{'n_estimators': [50, 100, 150, 250],
                    'max_depth': [10, 20, 40],
                    'max_features': (5, 10, 20, 30),
                    'max_leaf_nodes': [20, 40, 60]
                    }]

grid_search = GridSearchCV(iid = False,
                                estimator = final_model,
                                param_grid = parameters,
                                scoring = 'f1',
                                n_jobs = -1,
                                cv = 10)
grid_search.fit(x_train, y_train)

grid_search.best_score_
grid_search.best_params_