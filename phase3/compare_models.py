from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import(
accuracy_score,
precision_score,
recall_score,
f1_score,
confusion_matrix,
)

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

X = df.drop(columns = ["Result"])
y = df["Result"].map({-1: 1, 1: 0})

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42,
    eval_metric="logloss",
    verbosity=0,
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision :", precision_score(y_test, y_pred))
print("Recall :", recall_score(y_test, y_pred))
print("F1 :", f1_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))



