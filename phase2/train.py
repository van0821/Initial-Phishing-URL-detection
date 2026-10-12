import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
        .replace("b", "", regex=False)
        .replace("'", "", regex=False)
        .astype(int)
    )

# print(df.shape)
# print(df.dtypes)
# print(df["Result"].value_counts())

X = df.drop(columns=["Result"])
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# print(X_train.shape)
# print(X_test.shape)
# print(y_train.value_counts())
# print(y_test.value_counts())

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

same = sum(a == b for a, b in zip(y_pred, y_test))
Accuracy = accuracy_score(y_test, y_pred)
Precision = precision_score(y_test, y_pred, pos_label = -1)
Recall = recall_score(y_test, y_pred, pos_label = -1)
F1 = f1_score(y_test, y_pred, pos_label = -1)
ConfusionMatrix = confusion_matrix(y_test, y_pred, labels = [-1, 1])

print(f"Accuracy：{Accuracy:.2%}")
print(f"Precision：{Precision:.2%}")
print(f"Recall：{Recall:.2%}")
print(f"F1：{F1:.2%}")
print(ConfusionMatrix)

importance = model.feature_importances_
feature_names = X.columns

ranked = sorted(
    zip(feature_names, importance),
    key = lambda item: item[1],
    reverse = True,
)

for name, value in ranked[:10]:
    print(f"{name}: {value:.4f}")

top10 = ranked[:10]
names = [item[0] for item in top10]
values = [item[1] for item in top10]

plt.figure(figsize=(10, 6))
plt.barh(names, values)
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi = 150)
plt.show()

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    labels = [-1, 1],
    display_labels = ["钓鱼", "正常"],
    cmap = "Blues",
)

plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi = 150)
plt.show()


