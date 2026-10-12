# 钓鱼网站检测（Random Forest）

## 项目简介

本项目根据网页的 30 个特征，判断该网站是否为钓鱼网站。

## 数据集

- 数据来源：UCI Phishing Websites 数据集，`Training Dataset.arff`
- 样本数：11055
- 特征数：30
- 标签：`Result`
  - `-1` 表示钓鱼网站
  - `1` 表示正常网站

## 模型与选型

清洗后的数据是表格型数据，比较适合 Random Forest。

与 XGBoost 对比后发现，在当前参数和数据集下，Random Forest 的指标优于 XGBoost。

5 折交叉验证平均 Recall：

```text
Random Forest：95.57%
XGBoost      ：92.67%
```

## 评估结果

以下结果是 Random Forest 在 20% 测试集上的表现：

```text
Accuracy ：97.4%
Precision：98.0%
Recall   ：96.1%
F1       ：97.1%
```

## 项目结构

```text
data：共享数据集（`Training Dataset.arff`、特征说明文档）
phase1：Python 数据处理基础，pandas、matplotlib 练习
phase2：第一个机器学习项目，特征、标签、Random Forest、评估
phase3：模型对比、Flask 接口、模型保存与加载
```

## 使用方法

1. 训练并保存模型：

```bash
cd phase3
python save_model.py
```

2. 启动 Flask 接口：

```bash
cd phase3
python app.py
```

3. 调用接口：

```text
POST http://127.0.0.1:5000/predict
Content-Type: application/json
```

请求体示例：

```json
{"features":[-1,1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,0,1,1,1,1,-1,-1,-1,-1,1,1,-1]}
```

返回示例：

```json
{"prediction": -1, "label": "钓鱼"}
```

注意：`features` 数组必须按照训练时的 30 个特征顺序传入。

## 当前限制

- 接口接收的是 30 个已经提取好的特征，不是原始 URL
- 目前还没有实现“URL 字符串 → 30 个特征”的自动提取
- 标签含义：`-1` 为钓鱼，`1` 为正常
- XGBoost 对比时使用了 `0/1` 标签映射，与主模型的 `-1/1` 不同，需要区分
