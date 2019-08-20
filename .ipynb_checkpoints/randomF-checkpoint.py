#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 11:39:13 2017

@author: jpinzon
"""
########################################
#QUESTION # 2
########################################
#RANDOM FOREST MODEL McKENSSON
import os, sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_sql_query as rsq
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score

# Custom functions
def date_std (df):  
    # Transform data to delta-date
    df['DATE'] = pd.to_datetime(df['DATE']) 
    min_dt= min(df['DATE'])
    st_df = pd.DataFrame()
    for group in df.CUST_ID.unique():
        df1 = df[df['CUST_ID'] == group]
        df1['date_delta'] = (df1['DATE'] - min_dt) / np.timedelta64(1,'D')
        st_df = pd.concat([st_df, df1])
    st_df = st_df.drop(['DATE'],axis=1)
    return st_df

def run_model (model, train, test, predictors, predicted): 
    # Runs the specified model and calculates MSE and obb score
    model.fit(train[predictors],train[predicted])
    predictions = model.predict(test[predictors])
    MSE1 = mean_squared_error(predictions,test[predicted].values)
    MSE = "{0:6.2f}".format(MSE1)
    obb_score = "{0:.4f}".format(model.oob_score_)
    return float(MSE), float(obb_score)

def demandprediction(customer, price, model):
    # Predictd demand for a single customer
    df_demand=pd.DataFrame([[customer,price]],columns = n_predictors)
    demand=model.predict(df_demand)
    return round(demand[0])
    
def pred_per_customer(customer_list, price, model):
    # Generates a DF with predicion for all customers on a list
    df=pd.DataFrame()
    for customer in range(len(customer_list)):
        pred = demandprediction(customer,3, model)
        df2 = pd.DataFrame([[customer,price, pred]], columns=['CUST_ID','PRICE_USD','Pred_DEMAND']) 
        df=df.append(df2, ignore_index=True)
    return df


# READ DATA
os.chdir("/Users/jpinzon/Desktop/MCK_DS_Take_Home_Fall2017")

# From file
drug_a_sales = pd.read_csv('Dataset.csv')
# From SQL database
orders_db = sqlite3.connect('mk_orders_sp2017.db')
t_name = 'Spring_17_Drug_A'
drug_a_sales = rsq("Select * FROM "+t_name+";", orders_db)
drug_a_sales = drug_a_sales.drop(['index'], axis=1)

# Data Exploration
drug_a_sales.head(10)

drug_a_sales.describe()

# Visualization
drug_a_sales['PRICE_USD'].hist(bins=50)
drug_a_sales['ORDER_QTY'].hist(bins=50)

drug_a_sales.boxplot(column='PRICE_USD')
drug_a_sales.boxplot(column='ORDER_QTY')


drug_a_sales.boxplot(column='PRICE_USD', by = 'CUST_ID')
drug_a_sales.boxplot(column='ORDER_QTY', by = 'CUST_ID')

# Seems some demands are larger than the rest
for group in drug_a_sales['CUST_ID'].unique():
    df = drug_a_sales[drug_a_sales['CUST_ID']==group]
    plt.scatter(df['PRICE_USD'], df['ORDER_QTY'])

# REMOVE OUTLIERS
drug_a_sales['DATE'] = pd.to_datetime(drug_a_sales['DATE']) 
drug_a_sales = drug_a_sales.set_index('DATE')
drug_a_sales = drug_a_sales[drug_a_sales.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
drug_a_sales = drug_a_sales.reset_index()

drug_a_sales.boxplot(column='ORDER_QTY', by = 'CUST_ID')

for group in drug_a_sales['CUST_ID'].unique():
    df = drug_a_sales[drug_a_sales['CUST_ID']==group]
    plt.scatter(df['PRICE_USD'], df['ORDER_QTY'])


# Format DATE as Delta time for each group
drug_a_sales = date_std(drug_a_sales)

# Encoding the numeric columns as int64    
le = LabelEncoder()
for i in list(drug_a_sales.columns.values):
    drug_a_sales[i] = le.fit_transform(drug_a_sales[i])

drug_a_sales['CUST_ID'] = drug_a_sales['CUST_ID'].astype(str)

# Split the data into train and test datasets
df_train, df_test = train_test_split(drug_a_sales, test_size = 0.25)

# RANDOM FOREST REGRESSION
# Define features
predictors = ['CUST_ID','date_delta','PRICE_USD']
outcome = 'ORDER_QTY'
# CAN JUMP TO THE OPTIMAZED MODEL BELOW
# Raw Model
rf_reg_model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_split=1e-07, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=100, n_jobs=1, oob_score=True, random_state=None,
           verbose=0, warm_start=False)

scoring = run_model(rf_reg_model, df_train, df_test, predictors, outcome)
scoring

# MODEL OPTIOMIZATION
# Criteria
criteria=['mae', 'mse']
for crit in criteria:
    rf_reg_model = RandomForestRegressor(bootstrap=True, criterion=crit, 
                                   max_depth=None,max_features='auto', 
                                   max_leaf_nodes=None, min_impurity_split=1e-07,
                                   min_samples_leaf=1, min_samples_split=2,
                                   min_weight_fraction_leaf=0.0, n_estimators=100, 
                                   n_jobs=1, oob_score=True, random_state=None,
                                   verbose=0, warm_start=False)
    scoring = run_model(rf_reg_model, df_train, df_test, predictors, outcome)
    print (crit, '\n', scoring[0], scoring[1])
 
# Features
features=[0.3,0.5,1,2,3 ,'auto']
for feat in features:
    rf_reg_model = RandomForestRegressor(bootstrap=True, criterion='mse', 
                                   max_depth=None,max_features=feat, 
                                   max_leaf_nodes=None, min_impurity_split=1e-07,
                                   min_samples_leaf=1, min_samples_split=2,
                                   min_weight_fraction_leaf=0.0, n_estimators=100, 
                                   n_jobs=1, oob_score=True, random_state=None,
                                   verbose=0, warm_start=False)
    scoring = run_model(rf_reg_model, df_train, df_test, predictors, outcome)
    print (feat, '\n', scoring[0], scoring[1])
    
# n_estimators
estimators=[100, 500, 1000, 2000]
for est in estimators:
    rf_reg_model = RandomForestRegressor(bootstrap=True, criterion='mse', 
                                   max_depth=None,max_features='auto', 
                                   max_leaf_nodes=None, min_impurity_split=1e-07,
                                   min_samples_leaf=1, min_samples_split=2,
                                   min_weight_fraction_leaf=0.0, n_estimators=est, 
                                   n_jobs=1, oob_score=True, random_state=None,
                                   verbose=0, warm_start=False)
    scoring = run_model(rf_reg_model, df_train, df_test, predictors, outcome)
    print (est, '\n', scoring[0], scoring[1])

# n_jobs
jobs=[1,2,3, 4, 5]
for job in jobs:
    rf_reg_model = RandomForestRegressor(bootstrap=True, criterion='mse', 
                                   max_depth=None,max_features='auto', 
                                   max_leaf_nodes=None, min_impurity_split=1e-07,
                                   min_samples_leaf=1, min_samples_split=2,
                                   min_weight_fraction_leaf=0.0, n_estimators=100, 
                                   n_jobs=job, oob_score=True, random_state=None,
                                   verbose=0, warm_start=False)
    scoring = run_model(rf_reg_model, df_train, df_test, predictors, outcome)
    print (job, '\n', scoring[0], scoring[1])

 

# TRAINING OPTIMIZED MODEL
rf_reg_model_opt = RandomForestRegressor(criterion = 'mse', max_features = 'auto',
                                   n_estimators = 100,  n_jobs = 1, 
                                   oob_score = True, verbose = 0)
scoring = run_model(rf_reg_model_opt, df_train, df_test, predictors, outcome)
scoring

feature_imp_opt = pd.DataFrame(rf_reg_model_opt.feature_importances_, 
                           index=predictors).sort_values(by=[0],ascending=False)
feature_imp_opt

# CROSSVALIDATION
scores = cross_val_score(rf_reg_model_opt, df_train[predictors], df_train[outcome], cv=5)
mean_score = '{0:0.4f}'.format(scores.mean() )
mean_score

# PREDICTION MODEL 
# WITH NO DATE DATA - as it only accounts ~3%
rf_reg_model_opt_pred = rf_reg_model_opt
n_predictors=['CUST_ID', 'PRICE_USD']
scoring_pred = run_model(rf_reg_model_opt_pred, df_train, df_test, n_predictors, outcome)

feature_imp_pred = pd.DataFrame(rf_reg_model_opt_pred.feature_importances_, 
                                index=n_predictors).sort_values(by=[0],ascending=False)
feature_imp_pred

## PREDICTING DEMAND
all_cust = list(drug_a_sales['CUST_ID'].unique())
pred_per_customer(all_cust, 3, rf_reg_model_opt_pred)

########################################
#QUESTION # 3
########################################
# CLUSTERING USING K-MEANS
from sklearn.cluster import KMeans

# Determine meaningful groups or clusters:
clusters=[1,2,3,4,5,6,10, 25, 50, 100]
for cluster in clusters:
    kmeans = KMeans(n_clusters=cluster, random_state=0).fit(drug_a_sales)
    print(cluster, kmeans.score(drug_a_sales))

kmeans.score(drug_a_sales)
drug_a_sales['M_GROUP']=pd.DataFrame(kmeans.labels_)

# After creating the new dataset with a column indicating the meaningful groups.
# There are two strategies:
#
# 1. Built a model for each group
#    Determine the bets model for each group by dividing the dataset into subsets
#    each subset representing one group and built various models (e.g. logistic regression) 
#    with different algorithsm on each.
#    something like this:

algorithms = []
algorithms.append(('LR', LogisticRegression()))
algorithms.append(('NB', GaussianNB()))
algorithms.append(('SVM', SVC()))

# * requires to import all the modules. Other algorithms can be added. 

results = []
names = []
for name, algorithm in algorithms:
	kfold = KFold(n_splits=10, random_state=seed)
	cv_results = cross_val_score(algorithm, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
    
#    Pick the best for each group and perform the predictions for that group with the 
#    selected model.
# 2. Built a model for the whole dataset 
#    Basically will be building the same procedure I ran above with the supplied dataset, but
#    taking into consideration the new groups. 

