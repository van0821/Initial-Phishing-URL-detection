import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.io import arff
#读取arff文件
data, meta = arff.loadarff("../data/Training Dataset.arff")
#转换成DataFrame
df = pd.DataFrame(data)
#字符串转换成数字
for column in df.columns:
    df[column] = (
        df[column]
        .astype(str)
        .str.replace("b","",regex=False)
        .str.replace("'","",regex=False)
        .astype(int)
    )

print(df.head())
print(df.shape)
print(df.columns)
print(df.dtypes)
print(df["Result"].value_counts())
