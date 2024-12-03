import pandas as pd

# 读取CSV文件
file_path = "E:/datasets/UrbanSound8K/metadata/UrbanSound8K.csv"
df = pd.read_csv(file_path)

# 目标保留行数
target_rows = 3000

# 获取每个fold的数量
fold_counts = df['fold'].value_counts().sort_index()

# 每个fold大致保留的样本数
target_per_fold = target_rows // len(fold_counts)

# 按fold分组并采样
sampled_df = df.groupby('fold').apply(lambda x: x.sample(min(target_per_fold, len(x)), random_state=42))

# 重置索引
sampled_df.reset_index(drop=True, inplace=True)

# 修改classID和class列
sampled_df['classID'] = 0
sampled_df['class'] = "background"

# 保存为新CSV文件
sampled_df.to_csv("E:/datasets/UrbanSound8K/metadata/UrbanSound8K_sampled.csv", index=False)
print("Done!")
