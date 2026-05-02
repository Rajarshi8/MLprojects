import pandas as pd
import os

def load_wine_data():
    base_path = os.path.dirname(__file__) 

    red_path = os.path.join(base_path, "winequality-red.csv")
    white_path = os.path.join(base_path, "winequality-white.csv")

    red = pd.read_csv(red_path, sep=';')
    white = pd.read_csv(white_path, sep=';')

    red['type'] = 'red'
    white['type'] = 'white'

    df = pd.concat([red, white], ignore_index=True)
    return df