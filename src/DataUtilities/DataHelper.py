import os

import pandas
import xlrd as xlrd


def read_csv(path_to_csv: str=''):
    df = pandas.read_csv(path_to_csv, encoding='latin1')
    return df

def read_excel(path_to_excel: str):
    df = pandas.read_excel(xlrd.open_workbook(path_to_excel), engine='xlrd')
    return df
