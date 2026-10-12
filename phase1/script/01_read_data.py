import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from scipy.io import arff

data, meta = arff.loadarff("../../data/Training Dataset.arff")

df = pd.DataFrame(data)
# 将bytes转换成普通字符串
df = df.map(lambda x: x.decode() if isinstance(x, bytes) else x)
# 字符串转化为数字
df = df.apply(pd.to_numeric)

# print(df.dtypes)
# print(df.head())
# print(df.shape)
# print(df.columns)

# print("数据形状：")
# print(df.shape)
# print("\n列名：")
# print(df.columns.tolist())
# print("\n数据类型：")
# print(df.dtypes)
# print("\n基本信息：")
# print(df.info())
# print("\nresult分布：")
# print(df["Result"].value_counts())

# print(df)

print(f"URL_Length的最大值：{df["URL_Length"].max()}")
print(f"URL_Length的最小值：{df["URL_Length"].min()}")
print(f"URL_Length的平均值：{df["URL_Length"].mean()}")
#
print(df["having_IP_Address"].value_counts())
#
# counts = df["Result"].value_counts()
# print(f"Result = 1 有{counts.get(1,0)}条")
# print(f"Result = -1 有{counts.get(-1,0)}条")

print("\nURL长度与Result的交叉统计：")

table = pd.crosstab(
        df["URL_Length"],
        df["Result"],
        normalize = "index"
        )
table_renamed = table.rename(index={-1 : "Legitimate" , 0 : "Suspicious" , 1 : "Phishing"})
print(table_renamed)
table_renamed[1].plot.bar(rot=0,
                          title = "Phishing Rate by URL Length")
plt.ylabel("Phishing Rate")
plt.xlabel("URL Length Category")
plt.tight_layout()
plt.show()


