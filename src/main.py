import os
from configparser import ConfigParser

from src import TweetGrabber, ModelBuilder

config = ConfigParser()
config.read('../config.ini')

ModelBuilder.build_model()

entities = ['indiginous', 'development', 'human rights', 'crime', 'budget', 'housing']
people = ['Vivs4PDW', 'JaniceLukes', 'kevinkleinwpg', 'cindygilroy', 'B_MayesSTV']

people_tweets = []
for person in people:
    tweets = TweetGrabber.grab_tweets(person)
    people_tweets.append(tweets)
print('Done')