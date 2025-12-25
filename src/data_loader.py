import pandas as pd
from IPython.display import display
from .utils import get_path

def load_data(path=None):
    if path is None:
        path = get_path("data", "raw", "diabetes.csv")

    df = pd.read_csv(path)

    print("\n=== Dataset Preview ===")
    display(df.head())

    print("\n=== Dataset Info ===")
    print(df.info())

    print("\n=== Dataset Shape ===")
    print(df.shape)

    return df
