import pandas as pd

df = pd.read_json('/Users/mohanteotia/Desktop/mini_proj/IDS_NetML/Data/2_training_set.json.gz', lines=True)
print(df.head())
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")