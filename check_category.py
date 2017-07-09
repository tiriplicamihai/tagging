# Check that tweets in the same category have the same language.
from collections import defaultdict
import json
import random
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report


CATEGORIES = ['real_estate']


def main():
    for category in CATEGORIES:
        check_authors_vocabulary(category)

def check_authors_vocabulary(category):
    """Use 80% of the authors as training set and the rest as test set. A good score validates the
    assumption that there is a defined vocabulary for a given category.
    """
    print category
    with open('%s_tweets.json' % category, 'r') as f:
        tweets = json.load(f)

    tweets_by_author = defaultdict(list)
    for tweet in tweets['tweets']:
        tweets_by_author[tweet['author_name']].append(tweet)

    authors = tweets_by_author.keys()
    training_set_count = len(authors) * 80 / 100

    training_authors = random.sample(authors, training_set_count)
    test_authors = list(set(authors) - set(training_authors))

    train_set = []
    test_set = []
    for author, tweets in tweets_by_author.items():
        if author in training_authors:
            train_set.extend([prepare_tweet(t['text']) for t in tweets])
        else:
            test_set.extend([prepare_tweet(t['text']) for t in tweets])

    vectorizer = CountVectorizer(max_features=10000,
                                 #stop_words='english',
                                 max_df=0.7)
    classifier = OneClassSVM()
    text_clf = Pipeline([('vect', vectorizer),
                         ('tfidf', TfidfTransformer(sublinear_tf=True,
                                                    norm='l2')),
                         ('clf', classifier)])


    text_clf = text_clf.fit(train_set)

    predicted = text_clf.predict(test_set)

    print np.mean(predicted == 1)
    print classification_report([1 for _ in range(len(test_set))], predicted)


def prepare_tweet(text):
    return text.replace('#', '')

if __name__ == '__main__':
    main()
