import sklearn

from sklearn import linear_model
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


class ModelType:
    def __init__(self, vectorizer=None):
        self.vectorizer = vectorizer
        self.model = None

    def create_model(self):
        self.model = linear_model.LogisticRegression()
