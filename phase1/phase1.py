import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
matplotlib.rcParams["font.sans-serif"] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_data(path):
    "读取CSV文件"
    return pd.read_csv(path)

def add_length_column(df):
    "给数据加一列URL长度"
    df["length"] = df["url"].apply(len)
    return df

def filter_malicious(df):
    "筛选出恶意URL"
    return df[df["label"] == 1]

def plot_label_counts(df, output_path):
    "画柱状图并保存"
    counts = count_by_label(df)
    counts.plot(kind = "bar")
    plt.title("安全 VS 恶意 URL 数量")
    plt.savefig(output_path)
    plt.show()

def count_by_label(df):
    return df["label"].value_counts()

def plot_traffic_line(df, output_path):
    df.plot(x="day", y="requests", kind="line", marker="o")
    plt.title("一周请求量趋势")
    plt.savefig(output_path)
    plt.show()

if __name__ == "__main__":
    df = load_data("urls.csv")
    df = add_length_column(df)
    count_by_label(df)
    malicious = filter_malicious(df)
    print(malicious)

    traffic = pd.read_csv("traffic.csv")
    plot_traffic_line(traffic, "traffic_line.png")

    plot_label_counts(df, "label_count.png")


