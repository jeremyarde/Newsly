import os
from configparser import ConfigParser
from src import TweetGrabber, ModelBuilder

from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

stopwords = stopwords.words('english')

config = ConfigParser()
config.read('../config.ini')

# ModelBuilder.build_model()

entities = ['indiginous', 'development', 'human rights', 'crime', 'budget', 'housing']
people = ['Vivs4PDW', 'JaniceLukes', 'kevinkleinwpg', 'cindygilroy', 'B_MayesSTV']

people_tweets = []
for person in people:
    tweets = TweetGrabber.grab_tweets(person)
    people_tweets.append(tweets)


tweet_text_list = []
for person_tweets in people_tweets:
    person_tweet_text = []
    for tweet in person_tweets:
        person_tweet_text.append(tweet.get('tweet_text'))
    tweet_text_list.append(person_tweet_text)


tok = TfidfVectorizer(lowercase=True, analyzer='word', stop_words='english')
for person_tweets in tweet_text_list:
    tokenized = tok.fit_transform("".join(person_tweets))

print('Done')