import json
import sys

import tweepy

TWEET_COUNT = 200

def main(query):
    auth = tweepy.OAuthHandler("", "")
    api = tweepy.API(auth_handler=auth)

    tweets = list(tweepy.Cursor(api.search, q='%s -filter:retweets' % query, lang='en',
                                count=100, result_type='recent').items(10000))
    tweet_dataset = [tweet.text for tweet in tweets]
    import ipdb; ipdb.set_trace()

    filename = '_'.join(query.split())
    with open('%s_tweets.json' % filename, 'w') as f:
        json.dump({'tweets': tweet_dataset}, f)

if __name__ == '__main__':
    main(sys.argv[1])
