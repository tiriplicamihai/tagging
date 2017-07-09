# Check that tweets in the same category have the same language.
from collections import defaultdict
import json
import random
import numpy as np

from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report


CATEGORIES = [('education', 'education'), ('real_estate', 'real_estate'),
              ('government_organization', 'government'),
              ('health_medical_pharmacy', 'health'),
              ('shopping_retail', 'retail')]


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

             train_data.extend([prepare_tweet(t['text']) for t in training_tweets])
             train_labels.extend([category for _ in range(len(training_tweets))])

             test_data.extend([prepare_tweet(t['text']) for t in test_tweets])
             test_labels.extend([category for _ in range(len(test_tweets))])

    vectorizer = CountVectorizer(max_features=10000,
                                 stop_words='english',
                                 max_df=0.7)
    classifier = RandomForestClassifier(n_estimators=100)
    text_clf = Pipeline([('vect', vectorizer),
                         ('tfidf', TfidfTransformer(sublinear_tf=True,
                                                    norm='l2')),
                         ('clf', classifier)])


    print "Training model"
    text_clf = text_clf.fit(train_data, train_labels)

    joblib.dump(text_clf, 'tweet_clf.pkl')

    predicted = text_clf.predict(test_data)

    #print np.mean(predicted == test_targets)
    print classification_report(test_labels, predicted)


def prepare_tweet(text):
    return text.replace('#', '')

if __name__ == '__main__':
    main()
