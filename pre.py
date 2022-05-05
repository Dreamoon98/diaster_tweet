import re
import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from wordcloud import STOPWORDS
stop_words = set(stopwords.words('english')) | STOPWORDS

train = pd.read_csv("./data/train.csv",usecols=['id','text','target'])
test = pd.read_csv("./data/test.csv",usecols=['id','text'])

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)


def remove_user(text):
    user = re.compile(r'@\w+')
    return user.sub(r'', text)


def remove_number(text):
    number = re.compile(r'\d+')
    return number.sub(r'', text)


def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)


def remove_stopword(text):
    token = word_tokenize(text)
    new_token = [word for word in token if not word in stop_words]
    return TreebankWordDetokenizer().detokenize(new_token)


def pre(text, url=0, user=0, number=0, punct=0, stopword=0):
    if url == 1:
        text=remove_URL(text)
    if user == 1:
        text=remove_user(text)
    if number == 1:
        text=remove_number(text)
    if punct == 1:
        text=remove_punct(text)
    if stopword == 1:
        text=remove_stopword(text)
    return text


