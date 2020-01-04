#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''This script gets data from used cars, builds and tests several predictive models,
   and then makes price predictions on test data using the best model.'''


# # Import required libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Define data class

# In[ ]:


class Data:
    def __init__(self, train_file_path, test_file_path, target_var, num_features, cat_features):
        '''creates train, target, and test dataframes'''
        self.target_var = target_var
        self.train_df = self._clean_data(train_file_path, num_features, cat_features)
        self.test_df = self._clean_data(test_file_path, num_features, cat_features, outlier=False)
        self.target_df = self.train_df[self.target_var]
        self.num_features = [col for col in self.test_df.columns if self.test_df[col].dtype != 'object']
        self.cat_features = [col for col in self.test_df.columns if self.test_df[col].dtype == 'object']
        self.all_cols = self.num_features + self.cat_features + self.target_var        
        
    def target_summary(self):
        '''provides brief information and summary stats of the target variable'''
        print('\nSummary of the target variable:')
        print(self.train_df[target_var].info())
        print(self.train_df[target_var].describe())
    
    def train_summary(self):
        '''provides brief information and summary stats of the training data'''
        print('\nSummary of the training file:')
        train_cols = self.train_df.drop(columns=target_var).shape[1]
        train_rows = self.train_df.shape[0]
        print('\nThe train file has {} features and {} data-points'.format(train_cols, train_rows))
        print('\nData types in each column:')
        print(self.train_df.dtypes)
        print('\nNumber of missing values in each column:')
        print(self.train_df.isna().sum())
        print('\nFirst 3 rows of the training data:')
        print(self.train_df.head(3))
        print('\nNumber of unique values in each column:')
        print(self.train_df.nunique())
        print('\nDescriptive stats of numerical features:')
        train_num_cols = [col for col in self.train_df.columns if self.train_df[col].dtype != 'object']
        print(self.train_df[train_num_cols].describe())
        print('\nDescriptive stats of categorical features:')
        train_cat_cols = [col for col in self.train_df.columns if self.train_df[col].dtype == 'object']
        print(self.train_df[train_cat_cols].describe())
        
    def test_summary(self):
        '''provides brief information and summary stats of the test data'''
        print('\nSummary of the test file:')
        test_cols = self.test_df.shape[1]
        test_rows = self.test_df.shape[0]
        print('\nThe test file has {} features and {} data-points'.format(test_cols, test_rows))
        print('\nData types in each column:')
        print(self.test_df.dtypes)
        print('\nNumber of missing values in each column:')
        print(self.test_df.isna().sum())
        print('\nFirst 3 rows of the test data:')
        print(self.test_df.head(3))
        print('\nNumber of unique values in each columns:')
        print(self.test_df.nunique())
        print('\nDescriptive stats of numerical features:')
        test_num_cols = [col for col in self.test_df.columns if self.test_df[col].dtype != 'object']
        print(self.test_df[test_num_cols].describe())
        print('\nDescriptive stats of categorical features:')
        test_cat_cols = [col for col in self.test_df.columns if self.test_df[col].dtype == 'object']
        print(self.test_df[test_cat_cols].describe())        
        
    def visualize_target_var(self):
        '''creates plots of the target variable'''
        print('plotting the distribution of the target variable')
        plt.figure(figsize = (14, 4))
        plt.subplot(1, 2, 1)
        sns.boxplot(self.target_df)
        plt.title(self.target_var[0])
        plt.subplot(1, 2, 2)
        sns.distplot(self.target_df)
        plt.title(self.target_var[0])
        plt.show()
        
    def visualize_cat_features(self):
        '''creates plots of the categorical variables from the training set'''
        print('plotting the distribution of categorical variables of the training data')
        plt.figure(figsize=(14, 18))
        i = 1
        for col in self.cat_features:
            if self.train_df[col].nunique() < 15:
                plt.subplot(4, 2, i)
                sns.countplot(self.train_df[col])
                plt.title(col)
                plt.xticks(rotation=90)
                plt.tight_layout()                
                plt.subplot(4, 2, i+1)
                sns.boxplot(x=col, y=self.target_var[0], data=self.train_df)
                plt.title(col)
                plt.xticks(rotation=90)
                plt.tight_layout()
                i += 2
        plt.show()
        
    def visualize_num_features(self):
        '''creates plots of the numerical variables from the training set'''
        print('plotting the distribution and pairwise correlation of numerical variables in the training set')    
        sns.pairplot(self.train_df[self.num_features])
        plt.show()
        plt.figure(figsize=(16, 7))
        i = 1
        for col in self.num_features:
            plt.subplot(3, 3, i)
            sns.kdeplot(self.train_df[col], shade=True)
            plt.title(col)
            plt.tight_layout()
            i += 1
        plt.show()
        
    def plot_heatmap(self):
        '''plots heatmap of features and the target variable'''
        print('ploting heatmap to show correlations among variables')    
        plt.figure(figsize = (10, 9))
        sns.heatmap(self.train_df.corr(), cmap = 'Blues', annot = True, vmin = 0.2)
        plt.show()
    
    def _clean_data(self, file_path, num_features, cat_features, corrupt=True, missing=True, outlier=True):
        '''cleans the data by fixing corrupted, missing, and outlier values'''
        df = self._load_data(file_path)
        self.removed_cols = []
        for col in df.columns:
            if df[col].nunique()/len(df[col]) > 0.95:
                df = df.drop(columns=[col])
                self.removed_cols.append(col)
        # fix corrupt data
        if corrupt:
            for col in ['Mileage', 'Engine', 'Power', 'New_Price']:
                col_split = df[col].str.split(' ', expand=True)
                df[col] = col_split[0]
            df.replace(['null', 0], np.nan, inplace=True)
        # fix missing values
        if missing:
            df.dropna(thresh=5, inplace=True)
            for col in df.columns:
                if df[col].isna().sum()/len(df[col]) > 0.6:
                    df = df.drop(columns=[col])
                    self.removed_cols.append(col)
            for col in num_features:
                if col not in self.removed_cols:
                    df[col] = pd.to_numeric(df[col], errors='raise')
                    df[col].fillna(df[col].dropna().mean(), inplace=True)
            for col in cat_features:
                if col not in self.removed_cols:
                    df[col].fillna(df[col].mode(), inplace=True)                             
        # fix outlier data
        if outlier:
            for col in num_features:
                if col not in self.removed_cols:
                    pd.to_numeric(df[col])
                    min_range = np.min((df[col].describe()['mean'] - 5 * df[col].describe()['std']), 0)
                    max_range = df[col].describe()['mean'] + 5 * df[col].describe()['std']
                    df = df[(df[col] > np.min(min_range, 0)) & (df[col] < max_range)]       
        # return the cleaned dataframe
        return df
                
    def _load_data(self, file_path):
        return pd.read_csv(file_path)


# # Define feature engineering class

# In[ ]:


class FeatureEngineering:
    def __init__(self, data, encode=True, scale=False):
        '''performs feature engineering nad feature selection for modeling'''
        self.data = data
        if encode:
            self.data = self._feature_encode(data)
        if scale:
            self.data = self._feature_scale(data)
        
    def _feature_encode(self, data):
        dummy_cols = []
        for col in data.cat_features:
            # merge categorical features with low frequencies
            if data.train_df[col].nunique()/len(data.train_df[col]) < 0.1:
                for name, count in data.train_df[col].value_counts().items():
                    if count/len(data.train_df[col]) < 0.01:
                        data.train_df[col].replace(name, 'Rare', inplace=True)
            if data.test_df[col].nunique()/len(data.test_df[col]) < 0.1:
                for name, count in data.test_df[col].value_counts().items():
                    if count/len(data.test_df[col]) < 0.01:
                        data.test_df[col].replace(name, 'Rare', inplace=True)
            # target-encode categorical features with high number of unique values
            if data.train_df[col].nunique() > 10:
                from category_encoders.target_encoder import TargetEncoder
                encoder = TargetEncoder(cols=col)
                encoder.fit(data.train_df[col], data.train_df[data.target_var])
                data.train_df[col] = encoder.transform(data.train_df[col])
                data.test_df[col] = encoder.transform(data.test_df[col])
            else:
                dummy_cols.append(col)
        # create dummy variables from categorical features with low number of unique values
        data.train_df = pd.get_dummies(data.train_df, columns=dummy_cols, drop_first = True)
        data.test_df = pd.get_dummies(data.test_df, columns=dummy_cols, drop_first = True)
        data.target_df = data.train_df[data.target_var]
        
    def _feature_scale(self, data):
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            data.train_df = pd.DataFrame(scaler.fit_transform(data.train_df), columns=data.train_df.columns)
            data.test_df = pd.DataFrame(scaler.fit_transform(data.test_df), columns=data.test_df.columns)
            data.target_df = data.train_df[data.target_var]


# # Define modeling class

# In[ ]:


class Modeling:
    def __init__(self, models={}):
        '''inisializes data modeling'''
        self.models = models
        
    def add_model(self, model_name, model):
        '''adds new models for data modeling'''
        self.models[model_name] = model

    def modeling_summary(self, data, k_cross_val=3, processor=-1, metric='neg_mean_squared_error'):
        '''applies cross-validation and provides a summary of model performance'''
        self.k_cross_val = k_cross_val
        self.metric = metric
        self.scores = {}
        best_score = -99999
        from sklearn.model_selection import cross_val_score
        for model_name, model in self.models.items():
            score_list = cross_val_score(estimator=model,
                                         X=data.train_df.drop(columns=target_var),
                                         y=data.train_df[target_var],
                                         cv=k_cross_val,
                                         n_jobs=processor,
                                         scoring=self.metric)
            self.scores[model_name] = round(score_list.mean(), 3)
            if score_list.mean() > best_score:
                self.best_score = round(score_list.mean(), 3)
                self.best_model = model
                self.best_model_name = model_name
        self.best_model.fit(X=data.train_df.drop(columns=target_var), y=data.target_df)
        self.predictions = self.best_model.predict(data.test_df)
        print("Here is the list of applied models and their '{}' scores:\n {}".format(self.metric, self.scores))
        print("\nThe best model was '{}' with '{}' score of {}".format(self.best_model_name, self.metric, self.best_score))
            
    def get_feature_importance(self):
        '''returns sorted features based on their importances, calculated using RandomForest method'''
        from sklearn.ensemble import RandomForestRegressor
        features = data.train_df.drop(columns=target_var).columns
        model = RandomForestRegressor()
        model.fit(data.train_df.drop(columns=target_var), data.target_df)
        importances = model.feature_importances_
        feature_importances = pd.DataFrame({'Feature':features, 'Importance':importances})
        feature_importances.sort_values(by='Importance', ascending=False, inplace=True)
        feature_importances.set_index('Feature', inplace=True, drop=True)
        feature_importances[0:6].plot.bar()
        plt.show()
        return feature_importances
    
    def model_tuning(self, hyper_parameters):
        '''tunes the hyperparameters of the best model'''
        self.hyper_parameters = hyper_parameters
        from sklearn.model_selection import GridSearchCV
        grid_search = GridSearchCV(estimator=self.best_model,
                                   param_grid=self.hyper_parameters,
                                   scoring=self.metric,
                                   n_jobs=-1,
                                   cv=self.k_cross_val,
                                   return_train_score=True, refit=True)
        grid_search.fit(X=data.train_df.drop(columns=target_var), y=data.target_df)
        self.predictions = grid_search.predict(data.test_df)
        print('Best combination of hyperparameters are: {}'.format(grid_search.best_params_))
        print("Best achieved '{}' score is: {}".format(self.metric, round(grid_search.best_score_, 3)))
        
    def save_results(self, file_path):
        '''creates prediction and saves the results of the test set'''
        predict_file = pd.DataFrame({'Car_ID': data.test_df.index, 'Car_Price': self.predictions})
        predict_file.to_csv(file_path + '/Predicted_Price.csv', index=False)


# # Define parameters for creating models

# In[ ]:


train_file_path = 'C:/Users/mehdi/Downloads/used-cars-price-prediction/train-data.csv'
test_file_path = 'C:/Users/mehdi/Downloads/used-cars-price-prediction/test-data.csv'
target_var = ['Price']
num_features = ['Year', 'Kilometers_Driven', 'Seats', 'Mileage', 'Engine', 'Power', 'New_Price']
cat_features = ['Name', 'Location', 'Fuel_Type', 'Transmission', 'Owner_Type']


# # Create a Data object

# In[ ]:


data = Data(train_file_path, test_file_path, target_var, num_features, cat_features)


# # Get summary of data sets for EDA

# In[ ]:


data.target_summary()
data.train_summary()
data.test_summary()


# # Create visuals of data for EDA

# In[ ]:


data.visualize_target_var()
data.visualize_num_features()
data.visualize_cat_features()
data.plot_heatmap()


# # Apply feature engineering

# In[ ]:


myfeatures = FeatureEngineering(data)


# # Create and run models, then select the best one

# In[ ]:


mymodel = Modeling()

# add regression models
from sklearn.linear_model import LinearRegression
mymodel.add_model('lin_reg', LinearRegression())

from sklearn.ensemble import RandomForestRegressor
mymodel.add_model('rand_forest', RandomForestRegressor())

from sklearn.ensemble import GradientBoostingRegressor
mymodel.add_model('grad_boost', GradientBoostingRegressor())

mymodel.modeling_summary(data, metric='r2')


# # Tune the hyperparameters of the best model

# In[ ]:


hyper_parameters = [{'n_estimators': [40, 60, 100],
                     'max_depth': [5, 15, 40],
                     'min_samples_split': [40, 80, 100],
                     'max_features': [5, 8, 11]
                    }]
mymodel.model_tuning(hyper_parameters)


# # Get feature importances

# In[ ]:


mymodel.get_feature_importance()


# # Create predictions and save results of the test data

# In[ ]:


mymodel.save_results('C:/Users/mehdi/Downloads/used-cars-price-prediction')

