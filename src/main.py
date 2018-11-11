import os
from configparser import ConfigParser

import numpy as np
from numpy import unicode
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, fbeta_score, accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer

from src.DataUtilities import DataHelper
from src.Enums.SummarizerEnums import SummarizerType
from src.Models.TestModel import ModelType
from src.Summarizers.BaseSummarizer import BaseSummarizer
from src.Summarizers.SumySummarizer import SumySummarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

config = ConfigParser()
config.read('../config.ini')

#
# stemmer = Stemmer('english')
# tokenizer = Tokenizer('english')
#
# lsa = SumySummarizer(summarizerType=SummarizerType.LSA)
# ed = SumySummarizer(summarizerType=SummarizerType.Edmundson)
# lex = SumySummarizer(summarizerType=SummarizerType.LexRank)
# rand = SumySummarizer(summarizerType=SummarizerType.Random)
#
# url = "https://www.cbc.ca/news/canada/toronto/skinny-dipping-sharks-ripleys-1.4862945"
# url2 = "https://www.bbc.com/news/business-45986510"
#
# results = {'lsa': lsa.get_summary(url2),
#            'ed': ed.get_summary(url2),
#            'lex': lex.get_summary(url2),
#            'rand': rand.get_summary(url2)}
#
# print(results)
t = os.getcwd()

# df = DataHelper.read_excel(config['PATHS']['DataExcel'])

df = DataHelper.read_csv(config['PATHS']['DataCsv'])
df = df.dropna()  # drop all rows with nan
biases_unique = df['BIAS'].astype('U').unique()
classes = df['CLASS'].unique()

bias_classes = ['right, Least-biased', 'left', 'Left-center', 'Right-center']
# filter out the undesired labels:
df = df[df['BIAS'].isin(bias_classes)]

sources = df['SOURCE']
titles = df['TITLE']



# replace newlines ( replace with tokenization later...
text_data = df['TITLE'].str.replace('\n', '')
train_labels = df['BIAS'].str.replace('\n', '')

# text_list_unicode = [unicode(s, 'utf-8') for s in text_list]
text_data = text_data.values.astype('U')
vectorizer = TfidfVectorizer()
train_data = vectorizer.fit_transform(text_data)
train_labels = train_labels.astype('U')

x_train, x_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.3, random_state=1)

# # Grid search
# svc = SVC(gamma='scale')
# parameters = {'kernel':('linear', 'rbf'), 'C': [1, 10]}
# print('Starting Gridsearch...')
# clf = GridSearchCV(svc, parameters, cv=5, verbose=True, n_jobs=5)
# clf.fit(x_train, y_train)
# sorted(clf.cv_results_.keys())
# clf.score(x_test, y_test)

print("Linear SVC")
svc = LinearSVC(verbose=True)
svc.fit(X=x_train, y=y_train)
print(svc.coef_)
mean = svc.score(x_test, y_test)
mean_string = f"Mean accuracy: {mean}"
print(mean_string)


# Decision Tree
print("Decision Tree")
clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, x_train, y_train, cv=5)
print(scores.mean())


# Random Forrest
print("Random Forest")
clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, x_train, y_train, cv=5)
print(scores.mean())


print("")
