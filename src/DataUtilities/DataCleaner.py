import nltk

from src.DataUtilities import DataHelper
import pandas as pd

words = set(nltk.corpus.words.words())

class DataCleaner:
    def clean(self, csv_file_source, csv_path_target):
        df_dirty = DataHelper.get_data_from_source(csv_file_source)
        df_clean = DataCleaner._remove_nonsense_words(df_dirty)
        DataHelper.save_csv(df_clean, csv_path_target)

    @staticmethod
    def _remove_nonsense_words(df: pd.DataFrame):
        sentence = ""
        exclusions = ['SOURCE', 'CLASS', 'BIAS']
        for col in df.columns:
            if col not in exclusions:
                for index, contents in df[col].iteritems():
                    contents = contents.lower()
                    tokens = nltk.wordpunct_tokenize(contents)
                    sentence = [x for x in tokens if x in words]
                    sentence = " ".join(sentence)
                    df[col][index] = sentence
                    # sentence = contents.lower().join(w for w in nltk.wordpunct_tokenize(contents) if w in words or not w.isalpha())
        return df
