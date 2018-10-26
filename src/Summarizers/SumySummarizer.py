from typing import Type

from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as LSASumy
from sumy.summarizers.edmundson import EdmundsonSummarizer as EdSumy
from sumy.summarizers.lex_rank import LexRankSummarizer as LexRankSumy
from sumy.summarizers.random import RandomSummarizer as RandomSumy
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

from src.Enums.SummarizerEnums import Summarizer
from src.Summarizers.BaseSummarizer import BaseSummarizer


class SumySummarizer:
    def __init__(self, tokenizer=None, stemmer=None, summarizerType: Summarizer=Summarizer.LSA):
        self.Tokenizer = Tokenizer('english') if tokenizer is None else tokenizer
        self.Summarizer = None
        self.Stemmer = Stemmer('english') if stemmer is None else stemmer

        if summarizerType is Summarizer.LSA:
            self.Summarizer = LSASumy(self.Stemmer)
        elif summarizerType is Summarizer.Edmundson:
            self.Summarizer = EdSumy(self.Stemmer)
            self.Summarizer.bonus_words = ['Bonus']
            self.Summarizer.stigma_words = ['Stigma']
            self.Summarizer.null_words = ['Null']
        elif summarizerType is Summarizer.LexRank:
            self.Summarizer = LexRankSumy(self.Stemmer)
        elif summarizerType is Summarizer.Random:
            self.Summarizer = RandomSumy(self.Stemmer)
        else:
            raise Exception(f"{summarizerType}Summarizer type is not defined")

    def get_summary(self, text_source: str, num_sentences: int=5) -> []:
        # url = "https://www.cbc.ca/news/canada/toronto/skinny-dipping-sharks-ripleys-1.4862945"
        parser = HtmlParser.from_url(text_source, self.Tokenizer)

        doc = parser.document

        return self.Summarizer(doc, num_sentences)
