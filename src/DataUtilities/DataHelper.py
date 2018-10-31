import os

import pandas

def read_csv(path_to_csv: str=''):
    df = pandas.read_csv(path_to_csv)
    return df

    print("")

csv_to_dataframe()
