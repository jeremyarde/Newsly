from enum import Enum


class Summarizer(Enum):
    LSA = 1
    Edmundson = 2
    LexRank = 3
    Random = 4


class Stemmer(Enum):
    Sumy = 1


class Tokenizer(Enum):
    Sumy = 1
