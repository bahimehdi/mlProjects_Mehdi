import pandas as pd
try:
    df = pd.read_csv("annulation_hotel.csv")
    print(df.columns.tolist())
    print(df['Annule'].value_counts(normalize=True))
except Exception as e:
    print(e)
