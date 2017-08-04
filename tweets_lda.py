import random
import json
import re

import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from chowmein.label_finder import BigramLabelFinder
from chowmein.label_ranker import LabelRanker
from chowmein.pmi import PMICalculator
from chowmein.text import LabelCountVectorizer

n_features = 300
n_topics = 10
n_top_words = 25
label_min_df = 5
words_min_df = 5
n_labels = 10
n_cand_labels = 100
tag_constraints = [] #[('NN', 'NN'), ('JJ', 'NN')]

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic #%d:" % topic_idx
        print " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    print

def main():
    with open('machine_learning_tweets.json', 'r') as f:
        tweets = json.load(f)['tweets']

    clean_tweets = prepare_tweets(tweets)
    test_sample = random.sample(clean_tweets, int(0.1 * len(clean_tweets)))
    data_sample = list(set(clean_tweets) - set(test_sample))

    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')
    tf = tf_vectorizer.fit_transform(data_sample)
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
    lda.fit(tf)

    docs = [nltk.word_tokenize(doc) for doc in data_sample]
    finder = BigramLabelFinder('pmi', min_freq=label_min_df,
                               pos=tag_constraints)
    cand_labels = finder.find(docs, top_n=n_cand_labels)


    pmi_cal = PMICalculator(
        doc2word_vectorizer=CountVectorizer(max_df=0.95, min_df=5, max_features=n_features, stop_words='english'),
        doc2label_vectorizer=LabelCountVectorizer())
    pmi_w2l = pmi_cal.from_texts(docs, cand_labels)

    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)

    ranker = LabelRanker(apply_intra_topic_coverage=False)

    ranked_lables = ranker.top_k_labels(topic_models=lda.components_,
                                        pmi_w2l=pmi_w2l,
                                        index2label=pmi_cal.index2label_,
                                        label_models=None,
                                        k=n_labels)

    print 'Labels'
    print
    for i, labels in enumerate(ranked_lables):
        print(u"Topic {}: {}\n".format(i, ', '.join(map(lambda l: ' '.join(l), labels))))

def prepare_tweets(tweets):
    clean_tweets = []
    for tweet in tweets:
        clean_tweet = tweet.lower()
        clean_tweet = re.sub(r'https?:\/\/[a-zA-Z0-9\/\.]*', '', clean_tweet)
        clean_tweet = re.sub(r'"#(\w+)"', '', clean_tweet)
        clean_tweet = re.sub(r'"@(\w+)"', '', clean_tweet)
        clean_tweet = re.sub(r'machine learning', '', clean_tweet)
        clean_tweet = re.sub(r'machine-learning', '', clean_tweet)
        clean_tweet = re.sub(r'machinelearning', '', clean_tweet)
        clean_tweets.append(clean_tweet)

    return clean_tweets


if __name__ == '__main__':
    main()
