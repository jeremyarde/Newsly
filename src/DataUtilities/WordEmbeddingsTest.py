import gensim as gensim
from nltk.data import find

word2vec = str(find('models/word2vec_samples/pruned.word2vec.txt'))\

def get_embeddings(text_data):
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec, binary=False)
    return None