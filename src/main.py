import os
from configparser import ConfigParser

from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk import corpus

from src.DataUtilities import DataHelper, WordEmbeddingsTest
from src.Models import SklearnTest

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

WordEmbeddingsTest.get_embeddings(text_data)

# tfidf stuff
# vectorizer = TfidfVectorizer()
# tokenizer = Tokenizer(num_words=200)
#
# tokenizer.fit_on_texts(text_data.tolist())
# encoded_text_data = tokenizer.texts_to_matrix(text_data.tolist(), mode='count')
# train_data = encoded_text_data

# train_data = vectorizer.fit_transform(text_data)
train_labels = train_labels.astype('U')

x_train, x_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2, random_state=1)

predict_texts = ['trying to predict on a sentence', 'other sentence to try and predict on']
predict_texts = tokenizer.texts_to_matrix(predict_texts, mode='count')
SklearnTest.run_sklearn(x_train, y_train, x_test, y_test, predict_texts)


print("")
