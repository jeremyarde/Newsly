import sklearn

from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import train_test_split


class ModelType:
    def createmodel(self):
        model = linear_model.LogisticRegression()
