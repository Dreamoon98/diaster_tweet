import numpy as np
import pandas as pd
import re
from collections import defaultdict
import string
from nltk.corpus import stopwords
from wordcloud import STOPWORDS
import seaborn as sns
import matplotlib.pyplot as plt
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test_with_target.csv")
print(train.head())
disaster=0
nodisaster=0
null=0
for index,row in train.iterrows():
    if row['target'] == 1 :
        disaster+=1
    elif row['target'] == 0:
        nodisaster +=1



print(disaster)
print(nodisaster)
print(train.isnull().sum())
print(train[pd.isnull(train.keyword)])
key_null_train = train[pd.isnull(train.keyword)]
key_null_disaster=0
key_null_nodisaster=0
for index,row in key_null_train.iterrows():
    if row['target'] == 1 :
        key_null_disaster+=1
    elif row['target'] == 0:
        key_null_nodisaster +=1
print(key_null_disaster)
print(key_null_nodisaster)

disaster=0
nodisaster=0
key_null_disaster=0
key_null_nodisaster=0

for index,row in test.iterrows():
    if row['target'] == 1 :
        disaster+=1
    elif row['target'] == 0:
        nodisaster +=1

print(disaster)
print(nodisaster)
print(test.isnull().sum())

key_null_test = test[pd.isnull(test.keyword)]
key_null_disaster=0
key_null_nodisaster=0
for index,row in key_null_test.iterrows():
    if row['target'] == 1 :
        key_null_disaster+=1
    elif row['target'] == 0:
        key_null_nodisaster +=1
print(key_null_disaster)
print(key_null_nodisaster)
print(f'Number of unique values in keyword = {train["keyword"].nunique()} (Training) - {test["keyword"].nunique()} (Test)')
print(f'Number of unique values in location = {train["location"].nunique()} (Training) - {test["location"].nunique()} (Test)')
loc_null_train = train[pd.isnull(train.location)]
disaster=0
nodisaster=0
loc_null_disaster=0
loc_null_nodisaster=0
for index,row in loc_null_train.iterrows():
    if row['target'] == 1 :
        loc_null_disaster+=1
    elif row['target'] == 0:
        loc_null_nodisaster +=1
print(loc_null_disaster)
print(loc_null_nodisaster)

disaster=0
nodisaster=0
loc_null_disaster=0
loc_null_nodisaster=0

for index,row in test.iterrows():
    if row['target'] == 1 :
        disaster+=1
    elif row['target'] == 0:
        nodisaster +=1

print(disaster)
print(nodisaster)
print(test.isnull().sum())

loc_null_test = test[pd.isnull(test.location)]
loc_null_disaster=0
loc_null_nodisaster=0
for index,row in loc_null_test.iterrows():
    if row['target'] == 1 :
        loc_null_disaster+=1
    elif row['target'] == 0:
        loc_null_nodisaster +=1
print(loc_null_disaster)
print(loc_null_nodisaster)


