import os
import pandas
from configparser import ConfigParser

from nltk import RegexpTokenizer

from src import TweetGrabber, ModelBuilder

from collections import Counter
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# stopwords = stopwords.words('english')

config = ConfigParser()
config.read('../config.ini')

# ModelBuilder.build_model()

entities = ['indiginous', 'development', 'human rights', 'crime', 'budget', 'housing']
people = ['Vivs4PDW', 'JaniceLukes', 'kevinkleinwpg', 'cindygilroy', 'B_MayesSTV']
df = pandas.DataFrame()
tok = TfidfVectorizer(lowercase=True, analyzer='word', stop_words='english')

people_tweets = []
for person in people:
    tweets = TweetGrabber.grab_tweets(person)
    people_tweets.append(tweets)


tweet_text_list = []
for person_tweets in people_tweets:
    person_tweet_text = []
    for tweet in person_tweets:
        person_tweet_text.append(tweet.get('tweet_text'))

    # string_list = ["".join(x) for x in person_tweets]

    tokenizer = RegexpTokenizer(r'\w+')
    zen_no_punc = tokenizer.tokenize(' '.join(person_tweet_text))
    word_count_dict = Counter(w.title() for w in zen_no_punc if w.lower() not in stopwords.words())
    common = word_count_dict.most_common()

    word_freq_df = pandas.DataFrame.from_dict(common, columns=[person_tweets[0]['username']])

    df = pandas.concat([df, word_freq_df])


# remove_list = ['Https', 'Co', 'Rt', 'Amp']
df.to_csv("word_counts.csv")
