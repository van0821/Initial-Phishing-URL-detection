from scipy.io import arff
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


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

X = df.drop(columns="Result")
y = df["Result"].map({-1: 1, 1: 0})

rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(
    n_estimators=100,
    random_state=42,
    learning_rate=0.1,
    max_depth=4,
    eval_metric="logloss",
    verbosity=0,
)

for name, model in [("RandomForest", rf), ("XGBoost", xgb)]:
    scores = cross_val_score(model, X, y, cv=5, scoring="recall")
    print(name, "5折Recall: ", scores.round(4))
    print(name, "平均Recall: ", round(scores.mean(), 4))
    print("-" * 30)
