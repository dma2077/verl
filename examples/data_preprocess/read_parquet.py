import pandas as pd

# 读取parquet文件
file_path = '/llm_reco/dehua/data/food101/test.parquet'
df = pd.read_parquet(file_path)

# 显示前5行数据
print("\n前5行数据:")
print(df.head())

# 显示列名
print("\n列名:")
print(df.columns.tolist())

# 显示数据基本信息
print("\n数据基本信息:")
print(df.info())