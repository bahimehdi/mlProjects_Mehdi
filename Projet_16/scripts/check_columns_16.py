import pandas as pd
try:
    df = pd.read_csv("box_office.csv")
    print(df.columns.tolist())
    print(df.head())
    print(df.info())
except Exception as e:
    print(e)
