import os
from configparser import ConfigParser

from src import TweetGrabber

config = ConfigParser()
config.read('../config.ini')

# build_model()
TweetGrabber.grab_tweets("realDonaldTrump")
