import pandas as pd
import json
from pathlib import Path


base_path = Path(__file__).resolve().parent.parent

df = pd.read_csv(base_path/"data"/"mz2010_cf.csv", sep=';', header=0, low_memory=False)


with open(base_path/"notebooks"/"mappings.json", "r") as f:
    mappings = json.load(f)

selected_features = mappings["features"]
df = df[selected_features].copy()
df.rename(columns=mappings["feature_names"], inplace=True)


# transforming df for better understanding

df_labels = df.copy()

for column in df_labels.columns:
    if column in mappings:
        df_labels[column] = df_labels[column].map(mappings[column])

print(df_labels.head())


#saving df with and without numerical feature values.

df = df.apply(pd.to_numeric, errors="coerce")
df.to_csv(base_path/"data"/"data.csv", sep = ";", index = False)
df_labels.to_csv(base_path/"data"/"data_labels.csv", sep = ";", index = False)