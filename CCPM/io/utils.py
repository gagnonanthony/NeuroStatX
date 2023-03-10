import pandas as pd
import os


def load_df_in_any_format(file):
    """
    Load dataset in any .csv or .xlsx format.
    :param df:
    :return:
    """
    _, ext = os.path.splitext(file)
    if ext == '.csv':
        df = pd.read_csv(file)
    if ext == '.xlsx':
        df = pd.read_excel(file)

    return df

