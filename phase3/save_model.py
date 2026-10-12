import joblib
from scipy.io import arff
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

data, meta = arff.loadarff("../data/Training Dataset.arff")
df = pd.DataFrame(data)

for column in df.columns:
    df[column] = (
        df[column]
        .astype(str)
        .str.replace("b", "", regex=False)
        .str.replace("'", "", regex=False)
        .astype(int)
    )

X = df.drop(columns=["Result"])
y = df["Result"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(
    {
        "model": model,
        "feature_names": list(X.columns),
    },
    "phishing_rf.pkl",
)

print("模型已保存")
