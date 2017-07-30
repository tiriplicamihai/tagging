# Check that tweets in the same category have the same language.
from copy import deepcopy
from collections import defaultdict
import json
import random
from random import shuffle
from string import punctuation

import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
LabeledSentence = gensim.models.doc2vec.LabeledSentence # we'll talk about this down below
import numpy as np # high dimensional vector computing library.
import pandas as pd # provide sql-like data manipulation tools. very handy.
pd.options.mode.chained_assignment = None

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import scale

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout


n_dim = 200

CATEGORIES = [('education', 'education'), ('real_estate', 'real_estate'),
              ('government_organization', 'government'),
              ('health_medical_pharmacy', 'health'),
              ('shopping_retail', 'retail')]

CATEGORY_LABELS = {
    'education': 0,
    'real_estate': 1,
    'government': 2,
    'health': 3,
    'retail': 4,
}
INVERSE_CATEGORY_LABELS = {v: k for k, v in CATEGORY_LABELS.items()}


def main():
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for filename, category in CATEGORIES:
        with open('%s_tweets.json' % filename, 'r') as f:
            tweets = json.load(f)

        tweets_by_author = defaultdict(list)
        for tweet in tweets['tweets']:
            tweets_by_author[tweet['author_name']].append(tweet)

        for author, tweets in tweets_by_author.items():
             training_set_count = len(tweets) * 80 / 100
             training_tweets = tweets[:training_set_count]
             test_tweets = tweets[training_set_count:]

             train_data.extend([tokenize(t['text']) for t in training_tweets])
             train_labels.extend([category for _ in range(len(training_tweets))])

             test_data.extend([tokenize(t['text']) for t in test_tweets])
             test_labels.extend([category for _ in range(len(test_tweets))])

    train_data = [tokens for tokens in train_data if tokens != 'NC']
    test_data = [tokens for tokens in test_data if tokens != 'NC']

    train_data = labelize_tweets(train_data, 'TRAIN')
    test_data = labelize_tweets(test_data, 'TEST')

    tweet_w2v = Word2Vec(size=n_dim, min_count=10)
    tweet_w2v.build_vocab([tokens.words for tokens in tqdm(train_data)])
    tweet_w2v.train([tokens.words for tokens in tqdm(train_data)],
                    total_examples=len(train_data),
                    epochs=20)

    print 'building tf-idf matrix ...'
    vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
    matrix = vectorizer.fit_transform([x.words for x in train_data])
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    print 'vocab size :', len(tfidf)

    train_vecs_w2v = np.concatenate([build_word_vector(tweet_w2v, tfidf, z, n_dim) for z in tqdm(map(lambda x: x.words, train_data))])
    train_vecs_w2v = scale(train_vecs_w2v)

    test_vecs_w2v = np.concatenate([build_word_vector(tweet_w2v, tfidf, z, n_dim) for z in tqdm(map(lambda x: x.words, test_data))])
    test_vecs_w2v = scale(test_vecs_w2v)

    model = Sequential()
    model.add(Dense(128, activation='sigmoid', input_shape=(n_dim,)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='sigmoid',))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='sigmoid',))
    model.add(Dense(len(CATEGORY_LABELS), activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    y_train = [CATEGORY_LABELS[l] for l in train_labels]
    one_hot_labels = keras.utils.to_categorical(y_train, num_classes=len(CATEGORY_LABELS))
    model.fit(train_vecs_w2v, one_hot_labels, epochs=50, batch_size=32, verbose=2)

    y_test = [CATEGORY_LABELS[l] for l in test_labels]
    one_hot_test_labels = keras.utils.to_categorical(y_test, num_classes=len(CATEGORY_LABELS))
    score = model.evaluate(test_vecs_w2v, one_hot_test_labels, batch_size=128, verbose=2)
    print score

def transform_category(label):
    x = np.array([0 for _ in range(len(CATEGORY_LABELS))])
    x[label] = 1
    return x.reshape(1, -1)

def build_word_vector(tweet_w2v, tfidf, tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec


def tokenize(tweet):
    try:
        tweet = unicode(tweet.lower())
        tokens = tokenizer.tokenize(tweet)
        tokens = filter(lambda t: not t.startswith('@'), tokens)
        tokens = filter(lambda t: not t.startswith('#'), tokens)
        tokens = filter(lambda t: not t.startswith('http'), tokens)
        return tokens
    except:
        import ipdb; ipdb.set_trace()
        return 'NC'


def labelize_tweets(tweets, label_type):
    labelized = []
    for i,v in tqdm(enumerate(tweets)):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized


if __name__ == '__main__':
    main()
