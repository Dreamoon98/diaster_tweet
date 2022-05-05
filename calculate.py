import numpy as np
import pandas as pd
import re
import string
train = pd.read_csv("./data/train.csv",usecols=['id','text','target'])
test = pd.read_csv("./data/test_with_target.csv")

sub = pd.read_csv("./result/submission_bert_remove_user.csv")

sub['result'] = test['target']
tp=0
fp=0
tn=0
fn=0
print(sub.head())
print(sub.dtypes)

for index,row in sub.iterrows():

    if int(row['target']) == 1 :
        if int(row['result']) == 1:
            tp +=1
        elif int(row['result']) == 0:
            fp +=1
            print("fp" + str(row['id']))
    elif int(row['target']) == 0 :
        if int(row['result']) == 0:
            tn +=1
        elif int(row['result']) == 1:
            fn +=1
            print("fn" + str(row['id']))

print('fp' + str(fp))
print('fn' + str(fn))
accuracy = (tp+tn)/ (tp + tn + fp + fn)
precision = tp / (tp+fp)
recall = tp / (tp + fn)
f1=2*precision*recall/(precision+recall)

print('accuracy: ' + str(accuracy) + '\n'+'precision:' + str(precision)+ '\n'+'recall:' + str(recall)+ '\n'+'f1:' + str(f1)+ '\n')