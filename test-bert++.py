from transformers import AutoTokenizer,TFBertModel, BertModel,TFAutoModel,AutoModel
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from collections import defaultdict
import tensorflow as tf
from keras.layers import *
from keras.layers import GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from keras.losses import CategoricalCrossentropy,BinaryCrossentropy
from keras.metrics import CategoricalAccuracy,BinaryAccuracy
from pre import *
from keras.layers import Input, Dense
import keras_metrics as km
from pre_test import clean
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

train = pd.read_csv("./data/train.csv",usecols=['id','text','target'])
test = pd.read_csv("./data/test.csv",usecols=['id','text'])
submission = pd.read_csv("./data/sample_submission.csv")
tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')
bert = TFBertModel.from_pretrained('bert-large-uncased')
tok = AutoTokenizer.from_pretrained('./twitter/')
sb = TFAutoModel.from_pretrained("./twitter",from_pt=True)
#train['text']=train['text'].apply(lambda x : pre(x,url=0, user=1, number=0, punct=0, stopword = 0))
train['text']=train['text'].apply(lambda x : clean(x))
#loss: 0.3439 - accuracy: 0.8589


x_train = tokenizer(
#x_train = tok(

    text=train.text.tolist(),
    add_special_tokens=True,
    max_length=36,
    truncation=True,
    padding=True,
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)

y_train = train.target.values
input_ids = Input(shape=(36,), dtype=tf.int32, name="input_ids")
input_mask = Input(shape=(36,), dtype=tf.int32, name="attention_mask")
#embeddings = bert(input_ids, attention_mask=input_mask)[1]  # (0 is the last hidden states,1 means pooler_output)
out = bert(input_ids, attention_mask=input_mask)[0]  # (0 is the last hidden states,1 means pooler_output)

# out = tf.keras.layers.GlobalMaxPool1D()(embeddings)
'''
out = tf.keras.layers.Dropout(0.1)(embeddings)

out = Dense(128, activation='relu')(out)
out = tf.keras.layers.Dropout(0.1)(out)
out = Dense(32, activation='relu')(out)

y = Dense(1, activation='sigmoid')(out)
'''
#out = Dense(128, activation='relu')(embeddings)


out=(Bidirectional(LSTM(25,return_sequences=True)))(out)

out=Conv1D(30, 3, padding='same')(out)

out=GlobalMaxPooling1D()(out)
#out = out[:, 0, :]
y = Dense(1, activation='sigmoid')(out)
model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
model.layers[2].trainable = True

model.summary()
optimizer = Adam(
    learning_rate=6e-06, # this learning rate is for bert model.
    epsilon=1e-08,
    decay=0.01,
    clipnorm=1.0)

# Set loss and metrics
loss = BinaryCrossentropy(from_logits = True)
#metric = BinaryAccuracy('accuracy'),
metric=[km.f1_score(), km.binary_precision(), km.binary_recall(),BinaryAccuracy('accuracy')]
# Compile the model
model.compile(
    optimizer = optimizer,
    loss = loss,
    metrics = metric)

final = model.fit(
    x ={'input_ids':x_train['input_ids'],'attention_mask':x_train['attention_mask']} ,
    y = y_train,
#   validation_split = 0.1,
  epochs=9,
    batch_size=10#10
)
x_test = tokenizer(
#x_train = tok(

    text=test.text.tolist(),
    add_special_tokens=True,
    max_length=36,
    truncation=True,
    padding=True,
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)
predicted = model.predict({'input_ids':x_test['input_ids'],'attention_mask':x_test['attention_mask']})
y_predicted = np.where(predicted>0.5,1,0)
y_predicted = y_predicted.reshape((1,3263))[0]
sample_data = pd.DataFrame(columns=['id','target'])
sample_data['id'] = test.id
sample_data['target'] = y_predicted
sample_data.to_csv('submission_bert_manu',index = False)