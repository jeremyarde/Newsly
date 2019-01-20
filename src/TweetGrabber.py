import json

import tweepy

consumer_key = 'TnbeO1QHTVcDuF2ea7FAtKVni'
consumer_secret = '5rMfKo2NQX27SSR1dOmRgRfebIi4NTeDhC7zudEljgV434uAph'
access_token = '844779103-PWUCu60qshikDNyvLyINeUzLS7hFPVOVXK7bEUf9'
access_token_secret = 'HSOZA8JwveJXw7OA0ZX7p01TeZ4nc8AboIbuGQoU3kyqy'


def lambda_handler(event, context):
    # Getting the credentials to access the API
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

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
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    # Getting the api
    api = tweepy.API(auth)

    # Getting the tweets from the API then printing the text
    public_tweets = api.user_timeline(screen_name=username, count=200, include_rts=1, exclude_replies=True, trim_user=True)

    tweet_info = {
        'id': [],
        'username': [],
        'created': [],
        'tweet_text': [],
        "retweets": [],
        "users_mentioned": [],
        "has_urls":[],
        "retweeted_text": [],
    }
    for status in public_tweets:
        tweet_info.get("id").append(status.id)
        tweet_info.get('username').append(username),
        tweet_info.get('created').append(status.created_at),
        tweet_info.get('tweet_text').append(status.text),
        tweet_info.get("retweets").append(status.retweet_count),
        tweet_info.get("users_mentioned").append(True if len(status.entities.get('user_mentions')) > 0 else False),
        tweet_info.get("has_urls").append(True if len(status.entities.get('urls')) > 0 else False)
        tweet_info.get("retweeted_text").append(status.retweeted_status.text),
