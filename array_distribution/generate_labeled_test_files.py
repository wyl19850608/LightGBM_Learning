import pandas as pd
import numpy as np

# 设置随机种子，确保结果可复现
np.random.seed(42)

# 生成1000个用户ID
user_ids = [f"user_{i:04d}" for i in range(1, 1001)]

# 生成1000个label（0或1，大致各占一半）
labels = np.random.choice([0, 1], size=1000, p=[0.5, 0.5])

# 为file1生成score（0到1之间的随机小数）
scores_file1 = np.random.uniform(low=0.01, high=0.99, size=1000)
# 确保分数保留4位小数
scores_file1 = np.round(scores_file1, 4)

# 为file2生成不同的score（0到1之间的随机小数）
scores_file2 = np.random.uniform(low=0.01, high=0.99, size=1000)
scores_file2 = np.round(scores_file2, 4)

# 构建file1的数据
file1_data = {
    'user_id': user_ids,
    'score': scores_file1,
    'label': labels
}

# 构建file2的数据（user_id和label与file1完全相同，只有score不同）
file2_data = {
    'user_id': user_ids,  # 相同的user_id
    'score': scores_file2, # 不同的score
    'label': labels       # 相同的label
}

# 转换为DataFrame
df_file1 = pd.DataFrame(file1_data)
df_file2 = pd.DataFrame(file2_data)

# 保存为CSV文件，不包含索引列
df_file1.to_csv('file1.csv', index=False)
df_file2.to_csv('file2.csv', index=False)

# 验证数据一致性
print(f"已生成两个CSV文件，各包含1000行数据")
print(f"user_id一致性: {'相同' if (df_file1['user_id'].equals(df_file2['user_id'])) else '不同'}")
print(f"label一致性: {'相同' if (df_file1['label'].equals(df_file2['label'])) else '不同'}")
print(f"score差异性: {'不同' if not (df_file1['score'].equals(df_file2['score'])) else '相同'}")
print(f"label=0的数量: {sum(labels == 0)}")
print(f"label=1的数量: {sum(labels == 1)}")
