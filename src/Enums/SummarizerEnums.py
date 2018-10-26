from enum import Enum


class SummarizerType(Enum):
    LSA = 1
    Edmundson = 2
    LexRank = 3
    Random = 4


class StemmerType(Enum):
    Sumy = 1


class TokenizerType(Enum):
    Sumy = 1
