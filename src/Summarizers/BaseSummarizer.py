from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as LSASumy
from sumy.summarizers.edmundson import EdmundsonSummarizer as EdSumy
from sumy.summarizers.lex_rank import LexRankSummarizer as LexRankSumy
from sumy.summarizers.random import RandomSummarizer as RandomSumy
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words


class BaseSummarizer:
    def __init__(self, tokenizer=None, summarizer=None):
        self.Tokenizer = tokenizer
        self.Summarizer = summarizer

    def get_summary(self, text_source: str, num_sentences: int=5) -> []:
        self.Summarizer.get_summary(text_source, num_sentences)
