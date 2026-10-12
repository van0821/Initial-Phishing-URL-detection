import pandas as pd

titanic = pd.read_csv("titanic.csv")
# print(titanic.head())

ages = titanic["Age"]
# print(ages.head())
#print(titanic["Age"].shape)

age_sex = titanic[["Age","Sex"]]
# print(age_sex.head())
# print(type(titanic[["Age","Sex"]]))
# print(titanic[["Age","Sex"]].shape)

# 年龄大于35岁
above_35 = titanic[titanic["Age"] > 35]
# print(above_35.head())
# print(titanic["Age"]>35)
# print(above_35.shape)

# 泰坦尼克号二等舱和三等舱的乘客
class_23 = titanic[titanic["Pclass"].isin([2,3])]
# print(class_23.head())

# 处理已知年龄的乘客数据
age_no_na = titanic[titanic["Age"].notna()]
# print(age_no_na.head())
# print(age_no_na.shape)

# 选定特定的行和列
adult_name = titanic.loc[titanic["Age"]>35,"Name"]
print(adult_name.head())
print(titanic.iloc[9:25,2:5])