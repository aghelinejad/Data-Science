# Import libraries
import pandas as pd
import numpy as np

# Load the dataset in a dataframe object and include only four features
data_path = "creditcard.csv"
df = pd.read_csv(data_path)
df_train = df.drop('Time', axis=1)
features = df_train.iloc[:, :-1]
labels = df_train.iloc[:, -1]


# train-test splitting
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(features, labels,
                                                    test_size=0.3,
                                                    random_state=5)

# data preprocessing using SMOTE (Synthetic Minority Over-sampling Technique)
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=5, sampling_strategy='minority')
x_sm, y_sm = sm.fit_sample(x_train, y_train)

# data preprocessing using TomekLinks method (undersampling)
from imblearn.under_sampling import TomekLinks

tl = TomekLinks(sampling_strategy='majority')
x_tl, y_tl = tl.fit_sample(x_train, y_train)

# modeling: Random forest classifier
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(class_weight='balanced',
                                  n_estimators=50,
                                  criterion='entropy',
                                  max_depth=20,
                                  max_leaf_nodes=50,
                                  n_jobs=-1,
                                  random_state=10)

# calculate error metrics
from sklearn.metrics import precision_score, recall_score, f1_score


def get_metrics(data_type, train_features, train_labels):
    rf_model.fit(train_features, train_labels)
    predictions = rf_model.predict(x_test)

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    error_metrics = pd.DataFrame({'precision': [precision],
                                  'recall': [recall],
                                  'f1': [f1]}, index=[data_type])
    return error_metrics


metric_original = get_metrics('original', x_train, y_train)
metric_smote = get_metrics('smote', x_sm, y_sm)
metric_tomeklink = get_metrics('tomeklinke', x_tl, y_tl)

metrics = pd.DataFrame()
metrics = pd.concat([metric_original, metric_smote, metric_tomeklink])
print('Model Training Finished.\nCalculated error_metrics: \n{}'.format(metrics))

# Save model
import joblib

joblib.dump(rf_model, 'RFC_model.pkl')
print("Model dumped!")