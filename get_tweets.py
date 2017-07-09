import json
import sys

import tweepy

TWEET_COUNT = 200

def main(filename):
    with open(filename, 'r') as f:
        handlers = json.load(f)

    auth = tweepy.OAuthHandler("", "")
    api = tweepy.API(auth_handler=auth)
    tweet_dataset = []
    category = filename.split('.')[0]
    for handler in handlers['handlers']:
        try:
            tweets = api.user_timeline(handler, count=TWEET_COUNT)
        except Exception as e:
            print "Failed to get tweets for %s with error %s" % (handler, e)
            continue

        for tweet in tweets:
            if tweet.lang != 'en':
                continue
            tweet_dataset.append({
                'category': category,
                'text': tweet.text,
                'entities': tweet.entities,
                'author_name': tweet.author.name})

    with open('%s_tweets.json' % category, 'w') as f:
        json.dump({'tweets': tweet_dataset}, f)

if __name__ == '__main__':
    main(sys.argv[1])
