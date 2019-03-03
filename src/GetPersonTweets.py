import json

import pandas as pd

import TweetGrabber


df = pd.read_csv("../Data/Politicians/Winnipeg_Manitoba Politicians - Councillors.csv").fillna('')
df2 = pd.read_csv("../Data/Politicians/Winnipeg_Manitoba Politicians - MLAs.csv").fillna('')

person_tweets = []
for row in df.iterrows():
    info = row[1]
    person = info.get('TwitterHandle')
    try:
        tweets = TweetGrabber.grab_tweets(person)
        person_info = dict(
            name=info.Name,
            party=info.Party,
            position=info.Position,
            ward=info.Ward,
            twitter_handle=info.TwitterHandle,
            tweets=tweets,
        )
        person_tweets.append(person_info)
        print(f"Got tweets for: {person}")
        # person_tweets.append()
    except Exception as e:
        print(f"Could not get tweets for: {person}, issue: {e}")
        continue

for row in df2.iterrows():
    info = row[1]
    person = info.get('TwitterHandle')
    try:
        tweets = TweetGrabber.grab_tweets(person)
        person_info = dict(
            name=info.Name,
            party=info.Party,
            position="MLA",
            ward=info.Constituency,
            twitter_handle=info.TwitterHandle,
            tweets=tweets,
        )
        person_tweets.append(person_info)
        print(f"Got tweets for: {person}")
        # person_tweets.append()
    except Exception as e:
        print(f"Could not get tweets for: {person}, issue: {e}")
        continue


with open("tweets.json", "w") as f:
    json.dump(person_tweets, f, indent=4)
