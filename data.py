import numpy as np
import pandas as pd
import re
from collections import defaultdict
import string
from nltk.corpus import stopwords
from wordcloud import STOPWORDS
import seaborn as sns
import matplotlib.pyplot as plt
stop_words = set(stopwords.words('english')) | STOPWORDS
def create_corpus(data, target):
    corpus = []

    for x in data[data['target'] == target]['text'].str.lower().str.split():
        for i in x:
            corpus.append(i)

    return corpus

def top_10_stop_words(data):
    disaster_corpus = create_corpus(data, 1)
    not_disaster_corpus = create_corpus(data, 0)

    dic_disaster = defaultdict(int)
    dic_not_disaster = defaultdict(int)
    for word in disaster_corpus:
        if word in stop_words:
            dic_disaster[word] += 1
    for word in not_disaster_corpus:
        if word in stop_words:
            dic_not_disaster[word] += 1

    top_10_stopwords_disaster = sorted(dic_disaster.items(), key=lambda x: x[1], reverse=True)[:10]
    top_10_stopwords_not_disaster = sorted(dic_not_disaster.items(), key=lambda x: x[1], reverse=True)[:10]

    x1, y1 = zip(*top_10_stopwords_disaster)
    x2, y2 = zip(*top_10_stopwords_not_disaster)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.bar(x1, y1)
    ax1.set_title('Disaster Tweets')
    ax2.bar(x2, y2, color='#ff7f0e')
    ax2.set_title('Not Disaster Tweets')
    fig.suptitle('Top 10 Stop Words')
    plt.savefig('./pic/top_10_stop_words.png')
    #plt.show()


def top_10_hashtags(data):
    disaster_corpus = create_corpus(data, 1)
    not_disaster_corpus = create_corpus(data, 0)

    dic_disaster = defaultdict(int)
    dic_not_disaster = defaultdict(int)
    for word in disaster_corpus:
        if word.startswith('#'):
            dic_disaster[word] += 1
    for word in not_disaster_corpus:
        if word.startswith('#'):
            dic_not_disaster[word] += 1

    top_10_hashtags_disaster = sorted(dic_disaster.items(), key=lambda x: x[1], reverse=True)[:10]
    top_10_hashtags_not_disaster = sorted(dic_not_disaster.items(), key=lambda x: x[1], reverse=True)[:10]

    x1, y1 = zip(*top_10_hashtags_disaster)
    x2, y2 = zip(*top_10_hashtags_not_disaster)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.bar(x1, y1)
    ax1.set_title('Disaster Tweets')
    ax2.bar(x2, y2, color='#ff7f0e')
    ax2.set_title('Not Disaster Tweets')
    plt.gcf().autofmt_xdate()
    fig.suptitle('Top 10 Hashtags in Tweets')
    plt.savefig('./pic/top_10_hashtags.png')

    #plt.show()
def keyword_distribution_graph(data):
    plt.rcParams['figure.figsize'] = (7, 5)
    filtered = data[data.keyword.notnull()]

    disaster_tweets_with_keyword_count = filtered[filtered.target == 1].keyword.size
    not_disaster_tweets_with_keyword_count = filtered[filtered.target == 0].keyword.size
    plt.bar(10, disaster_tweets_with_keyword_count, 3, label="Disaster Tweets")
    plt.bar(15, not_disaster_tweets_with_keyword_count, 3, label="Not Disaster Tweets")
    plt.legend()
    plt.ylabel('Number of Examples')
    plt.title('Tweets with the Keyword Column')
    plt.show()

def punctuations_distribution(data):
    disaster_corpus = create_corpus(data, 1)
    not_disaster_corpus = create_corpus(data, 0)

    dic_disaster = defaultdict(int)
    dic_not_disaster = defaultdict(int)

    for item in (disaster_corpus):
        if item in string.punctuation:
            dic_disaster[item] += 1
    for item in (not_disaster_corpus):
        if item in string.punctuation:
            dic_not_disaster[item] += 1

    punctuations_distribution_disaster = sorted(dic_disaster.items(), key=lambda x: x[1], reverse=True)
    punctuations_distribution_not_disaster = sorted(dic_not_disaster.items(), key=lambda x: x[1], reverse=True)

    x1, y1 = zip(*punctuations_distribution_disaster)
    x2, y2 = zip(*punctuations_distribution_not_disaster)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.bar(x1, y1)
    ax1.set_title('Disaster Tweets')
    ax2.bar(x2, y2, color='#ff7f0e')
    ax2.set_title('Not Disaster Tweets')
    fig.suptitle('Punctuation in Tweets')
#    plt.show()
    plt.savefig('./pic/Punctuation in Tweets.png')

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
# word_count
train['word_count'] = train['text'].apply(lambda x: len(str(x).split()))
test['word_count'] =  test['text'].apply(lambda x: len(str(x).split()))

# char_count
train['char_count'] = train['text'].apply(lambda x: len(str(x)))
test['char_count'] = test['text'].apply(lambda x: len(str(x)))

#
train['mean_word_length'] = train['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test['mean_word_length'] = test['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

# url_count
train['url_count'] = train['text'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w or 'www' in w]))
test['url_count'] = test['text'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w or 'www' in w]))

# punctuation_count
train['punctuation_count'] = train['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
test['punctuation_count'] = test['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

# punctuation_count
train['number_count'] = train['text'].apply(lambda x: len([c for c in str(x) if c.isdigit() == True]))
test['number_count'] = test['text'].apply(lambda x: len([c for c in str(x) if c.isdigit() == True]))

# hashtag_count
train['hashtag_count'] = train['text'].apply(lambda x: len([c for c in str(x) if c == '#']))
test['hashtag_count'] = test['text'].apply(lambda x: len([c for c in str(x) if c == '#']))

# user_count
train['user_count'] = train['text'].apply(lambda x: len([c for c in str(x) if c == '@']))
test['user_count'] = test['text'].apply(lambda x: len([c for c in str(x) if c == '@']))

train['stop_word_count'] = train['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in stop_words]))
test['stop_word_count'] = test['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in stop_words]))

METAFEATURES = ['word_count', 'char_count', 'mean_word_length','url_count', 'punctuation_count', 'hashtag_count',
                'user_count', 'stop_word_count','number_count']
DISASTER_TWEETS = train['target'] == 1

fig, axes = plt.subplots(ncols=2, nrows=len(METAFEATURES), figsize=(20, 50), dpi=100)

for i, feature in enumerate(METAFEATURES):
    sns.distplot(train.loc[~DISASTER_TWEETS][feature], label='Not Disaster', ax=axes[i][0])
    sns.distplot(train.loc[DISASTER_TWEETS][feature], label='Disaster', ax=axes[i][0])

    sns.distplot(train[feature], label='Training', ax=axes[i][1])
    sns.distplot(test[feature], label='Test', ax=axes[i][1])

    for j in range(2):
        axes[i][j].set_xlabel('')
        axes[i][j].tick_params(axis='x', labelsize=12)
        axes[i][j].tick_params(axis='y', labelsize=12)
        axes[i][j].legend()

    axes[i][0].set_title(f'{feature} Target Distribution in Training Set', fontsize=13)
    axes[i][1].set_title(f'{feature} Training & Test Set Distribution', fontsize=13)

plt.savefig('./pic/details.png')

#plt.show()
top_10_stop_words(train)

top_10_hashtags(train)
punctuations_distribution(train)
