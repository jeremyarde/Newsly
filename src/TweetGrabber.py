import json

import tweepy

with open('../client_secret.json') as f:
    credentials = json.load(f)


def lambda_handler(event, context):
    # Getting the credentials to access the API
    auth = tweepy.OAuthHandler(credentials['consumer_key'], credentials['consumer_secret'])
    auth.set_access_token(credentials['access_token'], credentials['access_token_secret'])

    # Getting the api
    api = tweepy.API(auth)

    # Getting the tweets from the API then printing the text
    public_tweets = api.home_timeline()

    for tweet in public_tweets:
        print(tweet.text)

    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }


def grab_tweets(username: str):
    # Getting the credentials to access the API
    auth = tweepy.OAuthHandler(credentials['consumer_key'], credentials['consumer_secret'])
    auth.set_access_token(credentials['access_token'], credentials['access_token_secret'])

    # Getting the api
    api = tweepy.API(auth)

    # Getting the tweets from the API then printing the text
    public_tweets = api.user_timeline(screen_name=username, count=200, include_rts=1, exclude_replies=True, trim_user=True)

    all_tweets = []
    for status in public_tweets:
        tweet_info = {
            'id': status.id,
            'username': username,
            'created': status.created_at,
            'tweet_text': status.text,
            "retweets": status.retweet_count,
            "users_mentioned": True if len(status.entities.get('user_mentions')) > 0 else False,
            "has_urls": True if len(status.entities.get('urls')) > 0 else False,
        }

        all_tweets.append(tweet_info)

    return all_tweets
