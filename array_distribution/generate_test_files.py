import pandas as pd
import numpy as np

# 设置随机种子，确保结果可复现
np.random.seed(42)

# 生成共同用户ID（1000个）
common_user_ids = [f"user_{i:04d}" for i in range(1, 1001)]

# 生成各自独有的用户ID（每个文件500个）
file1_unique_users = [f"file1_user_{i:04d}" for i in range(1, 501)]
file2_unique_users = [f"file2_user_{i:04d}" for i in range(1, 501)]

# 为共同用户生成分数（file1的分数整体略高于file2）
# file1的分数：均值0.7，标准差0.15，范围[0, 1]
common_scores_file1 = np.clip(np.random.normal(0.7, 0.15, 1000), 0, 1)
# file2的分数：均值0.65，标准差0.15，范围[0, 1]
common_scores_file2 = np.clip(np.random.normal(0.65, 0.15, 1000), 0, 1)

# 为独有用户生成分数
# file1独有用户：均值0.6，标准差0.2
file1_unique_scores = np.clip(np.random.normal(0.6, 0.2, 500), 0, 1)
# file2独有用户：均值0.75，标准差0.18
file2_unique_scores = np.clip(np.random.normal(0.75, 0.18, 500), 0, 1)

# 构建file1的数据
file1_data = {
    'user_id': common_user_ids + file1_unique_users,
    'score': list(common_scores_file1) + list(file1_unique_scores)
}

# 构建file2的数据
file2_data = {
    'user_id': common_user_ids + file2_unique_users,
    'score': list(common_scores_file2) + list(file2_unique_scores)
}

# 转换为DataFrame
df_file1 = pd.DataFrame(file1_data)
df_file2 = pd.DataFrame(file2_data)

# 保存为CSV文件
df_file1.to_csv('file1.csv', index=False)
df_file2.to_csv('file2.csv', index=False)

print(f"已生成测试文件:")
print(f"file1.csv - 包含 {len(df_file1)} 条记录")
print(f"file2.csv - 包含 {len(df_file2)} 条记录")
print(f"共同用户数量: {len(common_user_ids)}")
