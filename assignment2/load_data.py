import pandas as pd
import os

def load_adult_data():
    cols = [
        "age","workclass","fnlwgt","education","education_num",
        "marital_status","occupation","relationship","race","sex",
        "capital_gain","capital_loss","hours_per_week","native_country","income"
    ]

    base_path = os.path.dirname(__file__)

    file_path = os.path.join(base_path, "adult.csv")  # correct filename

    df = pd.read_csv(file_path, names=cols, na_values=" ?", skipinitialspace=True)
    return df