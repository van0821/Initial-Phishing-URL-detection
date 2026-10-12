import pandas as pd

df = pd.read_csv("urls.csv")

print(df)
print("---")
print(df["label"].value_counts())

# print(type(df))
