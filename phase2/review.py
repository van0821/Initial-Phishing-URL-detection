from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

data, meta = arff.loadarff("../data/Training Dataset.arff")
df = pd.DataFrame(data)

# 把 b'-1' 这类字节数据转成整数
for column in df.columns:
    df[column] = (
        df[column]
        .astype(str)
        .str.replace("b", "", regex=False)
        .str.replace("'", "", regex=False)
        .astype(int)
    )

# 特征 X 和标签 y
X = df.drop(columns=["Result"])
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42,
    stratify = y,
)

model = RandomForestClassifier(
    n_estimators = 100,
    random_state = 42,
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, pos_label=-1))
print("Recall   :", recall_score(y_test, y_pred, pos_label=-1))
print("F1       :", f1_score(y_test, y_pred, pos_label=-1))
print(confusion_matrix(y_test, y_pred, labels=[-1, 1]))
