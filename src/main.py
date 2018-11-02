import os
from configparser import ConfigParser

import numpy as np
from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer

from src.DataUtilities import DataHelper
from src.Enums.SummarizerEnums import SummarizerType
from src.Summarizers.BaseSummarizer import BaseSummarizer
from src.Summarizers.SumySummarizer import SumySummarizer

config = ConfigParser()
config.read('config.ini')

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

unique = df['BIAS'].apply(sorted, axis=1).unique()
unique1 = df['CLASS'].unique()
unique2 = df['SOURCE'].unique()

print("")
