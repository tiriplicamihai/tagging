from collections import defaultdict
import json
import os
import re

import chardet
import numpy as np
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

from extract_categories import CATEGORIES_FILE
from nltk.stem.snowball import SnowballStemmer
import random


DATA_SET_PATH = './techtc300'


def main():
    """Train a probabilistic classifier for each category and save it. """

    category_to_ids = _get_categories()

    categories_mapping = {}
    for count, category in enumerate(category_to_ids.keys()):
        categories_mapping[category] = count

    with open('categories_mapping.json', 'w') as f:
        json.dump(categories_mapping, f)

    train_data = []
    train_targets = []
    test_data = []
    test_targets = []

    for category, ids in category_to_ids.items():
        print category
        train_pos_data, test_pos_data = _read_data(ids[:5])

        train_data.extend(train_pos_data)
        train_targets.extend([categories_mapping[category] for _ in range(len(train_pos_data))])

        test_data.extend(test_pos_data)
        test_targets.extend([categories_mapping[category] for _ in range(len(test_pos_data))])

    # classifier = SGDClassifier(loss='modified_huber', penalty='l2',
    #                            alpha=1e-3, n_iter=10, random_state=42)
    vectorizer = CountVectorizer(max_features=10000,
                                 stop_words='english',
                                 max_df=0.7,
                                 ngram_range=(1,3))
    classifier = RandomForestClassifier(n_estimators=100)
    text_clf = Pipeline([('vect', vectorizer),
                         ('tfidf', TfidfTransformer(sublinear_tf=True,
                                                    norm='l2')),
                         ('clf', classifier)])


    print "Training model"
    text_clf = text_clf.fit(train_data, train_targets)

    joblib.dump(text_clf, 'random_forst_clf.pkl')

    predicted = text_clf.predict(test_data)

    print np.mean(predicted == test_targets)
    print(classification_report(test_targets, predicted, target_names=category_to_ids.keys()))

def extract_category_features():
    keywords  = {}
    text_clf = joblib.load('clf.pkl')

    category_to_ids = _get_categories()

    for category, ids in category_to_ids.items():
        ngrams = {}
        print category
        content, test = _read_data(ids[:5])
        content.extend(test)

        count_vect = text_clf.named_steps['vect']
        matrix = count_vect.fit_transform(content)
        freqs = [(word, matrix.getcol(idx).sum()) for word, idx in count_vect.vocabulary_.items()]
        sorted_keywords = sorted(freqs, key = lambda x: -x[1])
        ngrams['onegram'] = filter(lambda x: len(x[0].split(" ")) is 1, sorted_keywords)[:50]
        ngrams['twogram'] = filter(lambda x: len(x[0].split(" ")) is 2, sorted_keywords)[:50]
        ngrams['threegram'] = filter(lambda x: len(x[0].split(" ")) is 3, sorted_keywords)[:50]
        keywords[category] = ngrams

    with open('categories_keywords.json', 'w') as f:
        json.dump(keywords, f)


def _get_categories():
    with open('train_categories.json', 'r') as f:
        id_category_mapping = json.load(f)

    # Revert categories because multiple ids point to the same one.
    category_to_id_mapping = defaultdict(list)

    for id, category in id_category_mapping.items():
        if category.startswith("Regional"):
            continue

        category_to_id_mapping[category].append(id)

    return category_to_id_mapping


def _read_data(ids):
    """Load data from the dataset folders, format it and split it into test and data set - 80% for
    training and 20% for testing.
    """
    positive_docs = []
    negative_docs = []
    for root, dir, files in os.walk(DATA_SET_PATH):
        if not any(id in root for id in ids):
            # Does not contain any data for these ids.
            continue

        # Keep the id that matched the folder name
        matches = [id for id in ids if id in root]
        # Get the position of the id that matched the folder name.
        # 1 position = pos example
        # 2 position = neg example
        position = [root.split("_").index(id) for id in matches][0]

        if position is 1:
            path = '%s/all_pos.txt' % root
        if position is 2:
            path = '%s/all_neg.txt' % root

        positive_docs.extend(_read_docs(path))

    pos_train_count = len(positive_docs) * 8 / 10

    return (positive_docs[:pos_train_count], positive_docs[pos_train_count:])


def _read_docs(path):
    """Extract docs from dataset files. """
    with open(path, 'r') as f:
        lines = f.readlines()

    # Doc starts with two lines: "<dmoz_doc>" "id=0"
    i = 2
    lines_no = len(lines)
    docs = []
    while i < lines_no:
        if lines[i].startswith('<dmoz_subdoc>'):
            # Starts a new doc.
            doc = ""
            i += 1
            while i < lines_no and not lines[i].startswith('</dmoz_subdoc>'):
                doc = doc + lines[i]
                i += 1

            doc = doc.replace('\r\n', ' ')

            # Collapse spaces
            regex = re.sub(r'[ \t\n]+', ' ', doc)

            # Remove unnecessary annotations.
            doc = re.sub(r'\[\w+\.\w+\]', '', doc)

            charset = chardet.detect(doc).get('encoding') or 'utf-8'
            try:
                doc = doc.decode(charset)
                docs.append(doc)
            except UnicodeDecodeError:
                print 'Failed to decode doc %s.' % path
                return []

        i += 1

    return docs

if __name__ == '__main__':
    main()
    #extract_category_features()
