import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pprint
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 提取url特征的函数
def extract_features(url) -> dict:
    URL_Length = len(url)           #url长度
    has_at = int('@' in url)        #是否包含@
    has_https = int(url.startswith("https"))    #是否为https开头
    dot_counts = url.count('.')                 # .的数量
    hyphen_count = url.count('-')               #连字符的数量
    slash_count = url.count("/")                #反斜杠的数量
    question_count = url.count("?")             #问号的数量

    result = urlparse(url)                      #拆开url
    parts = result.netloc.split('.')            #存储网络位置
    if len(parts) <= 2:                         #判断子域名的数量
        subdomain_count = 0                     #如果域名数量小于2，子域名为0
    else:
        subdomain_count = len(parts) - 2        #否则子域名数量为去掉主域名和顶级域名后的数量（-2）

    domain_length = len(result.netloc)          #域名长度
    path_length = len(result.path)              #路径长度
    query_length = len(result.query)            #参数长度
    if len(result.query) == 0:                  #如果没有参数则为0
        query_count = 0
    else:                                       #如果有参数则分割'&'字符串，统计数量
        query_count = len(result.query.split('&'))

    return {"URL_Length": URL_Length,
            "has_at": has_at,
            "has_https": has_https,
            "dot_counts": dot_counts,
            "hyphen_count": hyphen_count,
            "slash_count": slash_count,
            "question_count": question_count,
            "subdomain_count": subdomain_count,
            "domain_length": domain_length,
            "path_length": path_length,
            "query_length": query_length,
            "query_count": query_count
            }

# url = "http://example.com/login"
# features = extract_features(url)
# pprint.pprint(features)

# url1 = "http://example.com/login"
# url2 = "https://www.google.com"
# url3 = "http://www.pay-pal-login.com@evil.com"
#
# print(extract_features(url1))
# print(extract_features(url2))
# print(extract_features(url3))

# url = "https://login.example.com/user?id=123"
#
# result = urlparse(url)
#
# domain = result.netloc
# parts = domain.split(".")
# print(parts)
# print(len(parts))
#
# #
# print(result)
# print("协议：", result.scheme)
# print("域名：", result.netloc)
# print("路径：", result.path)
# print("参数：", result.query)
# 以上为字符串特征提取练习

urls = [
    ("https://www.baidu.com", 0),
    ("https://www.google.com", 0),
    ("https://github.com", 0),
    ("https://www.microsoft.com", 0),
    ("https://www.apple.com", 0),

    ("http://paypal-login.com@evil.com", 1),
    ("http://secure-login-account.com", 1),
    ("http://login.account.verify.example.com", 1),
    ("http://example.com/login?user=admin&password=123", 1),
    ("http://www.pay-pal-security.com/login", 1)
]

data = []
for url, label in urls:
    features = extract_features(url)
    row = {"url": url}
    row.update(features)
    row["label"] = label
    data.append(row)

df = pd.DataFrame(data)
# print(df)
# print(df.shape)
# print(df.columns)
# print(df.dtypes)
#
# grouped = df.groupby("label").mean(numeric_only = True)
# print(grouped)
# print(grouped.T)
# print(df)

# 测试数据
# url = "http://secure-login-account.com/login/user?id=123&admin=true"
# test = urlparse(url)
#print(f"domain_length: {domain_length}, path_length: {path_length}, query_count: {query_count}")

X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state = 42
)

# print("X_train:", X_train.shape)
# print("X_test:", X_test.shape)
# print("y_train:", y_train.shape)
# print("y_test:", y_test.shape)

model = RandomForestClassifier(
    n_estimators = 100,
    random_state = 42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("真实标签：", y_test.values)
print("预测标签：", y_pred)

same = sum(a == b for a, b in zip(y_test, y_pred))
Accurancy = same / len(y_test)
print(f"准确率：{Accurancy:.2%}")

from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1_score:", f1_score(y_test, y_pred))
print("混淆矩阵:")
print(confusion_matrix(y_test, y_pred))
