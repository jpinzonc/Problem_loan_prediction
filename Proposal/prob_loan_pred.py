#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 14:36:47 2017

@author: jpinzon
"""

os.listdir()
import pandas as pd

data_1Q=pd.read_csv('LoanStats_2017Q1.csv', skiprows=1)
data_2Q=pd.read_csv('LoanStats3a.csv', skiprows=1)

data_1Q.shape[0]+data_2Q.shape[0]


# Merging the quaters into one 
dataframes = [data_1Q, data_2Q]
data = pd.concat(dataframes)
#removing last two columns that had non relevant information
data = data.sort_values(by = 'id',na_position = 'first', ascending=True)
data.tail(15)
data.drop(data.tail(5).index,inplace=True)
data.shape
data.id.unique()
data.to_csv('loans_2017.csv', index=False)

data_2017 = pd.read_csv('loans_2017.csv')



pd.DataFrame(a.columns)


a = data_2017.dropna(axis=1, how='all')
a.shape
a.loan_status.unique()
# Removing fully paid loans
a = a[a.loan_status.str.contains('Current')==False]
a.loan_status=a.loan_status.str.replace(r'(^.*Paid$)', "Fully Paid")
a.loan_status=a.loan_status.str.replace(r'(^.*Charged Off$)', "Charged Off")

a.settlement_amount.unique()

usef_cols = pd.DataFrame(a.isnull().sum().sort_values([a.isnull().sum().sort_values()<20000])
usef_cols.reset_index(drop = False, inplace=True)
usef_cols.columns = (['col_name','num_NaN'])
new_data =pd.DataFrame()

for i in range(0,len(usef_cols)):#(usef_cols.col_name):
    print(i)
    j = i+1
    df = a.iloc[:,i:j]
    print(df.head)
    new_data = pd.concat([new_data, df], axis=1)
    
new_data.head() 

new_data = pd.DataFrame(new_data.isnull().sum().sort_values()[a.isnull().sum().sort_values()<200])
a.total_acc

a[bbb]
a[usef_cols.col_name[0]]

type(usef_cols.col_name[0])