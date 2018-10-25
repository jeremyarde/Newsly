from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as LSASumy
from sumy.summarizers.edmundson import EdmundsonSummarizer as EdSumy
from sumy.summarizers.lex_rank import LexRankSummarizer as LexRankSumy
from sumy.summarizers.random import RandomSummarizer as RandomSumy
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words


class SumySummarizer:
    def __init__(self, stemmer=None, tokenizer=None, summarizer=None):
        self.Stemmer = stemmer
        self.Tokenizer = tokenizer
        self.Summarizer = summarizer

    @staticmethod
    def get_summary() -> []:
        stemmer = Stemmer('english')
        lsa = LSASumy(stemmer)
        ed = EdSumy(stemmer)
        lex = LexRankSumy(stemmer)
        rand = RandomSumy(stemmer)

        url = "https://www.cbc.ca/news/canada/toronto/skinny-dipping-sharks-ripleys-1.4862945"
        parser = HtmlParser.from_url(url, Tokenizer('english'))
        doc = parser.document

        results = {'lsa': lsa(doc, 5),
                   'ed': ed(doc, 5),
                   'lex': lex(doc, 5),
                   'rand': rand(doc, 5)}
        # lsatext = lsa(doc, 5)
        # edt = ed(doc, 5)
        # lext = lex(doc, 5)
        # randt = rand(doc, 5)

        for key in results:
            print(results.get(key))
