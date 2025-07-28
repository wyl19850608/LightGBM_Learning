import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 示例数据 - 两组模型输出的userid:分数
model1_scores = [
    {"userid": "user1", "score": 0.85},
    {"userid": "user2", "score": 0.72},
    {"userid": "user3", "score": 0.91},
    {"userid": "user4", "score": 0.68},
    {"userid": "user5", "score": 0.79},
    {"userid": "user6", "score": 0.88}  # 仅在模型1中存在
]

model2_scores = [
    {"userid": "user1", "score": 0.82},
    {"userid": "user2", "score": 0.75},
    {"userid": "user3", "score": 0.89},
    {"userid": "user4", "score": 0.71},
    {"userid": "user5", "score": 0.83},
    {"userid": "user7", "score": 0.65}  # 仅在模型2中存在
]

# 转换为DataFrame以便处理
df1 = pd.DataFrame(model1_scores).set_index('userid')
df2 = pd.DataFrame(model2_scores).set_index('userid')

# 合并两个DataFrame，对齐userid
comparison = pd.merge(
    df1, df2,
    left_index=True, right_index=True,
    how='outer',  # 保留所有userid
    suffixes=('_model1', '_model2')
)

# 计算分数差异
comparison['difference'] = comparison['score_model1'] - comparison['score_model2']

# 显示完整对比表格
print("用户分数对比表:")
print(comparison.round(4))  # 保留四位小数

# 筛选出两个模型都存在的用户
common_users = comparison.dropna()

# 可视化差异
plt.figure(figsize=(12, 6))

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 绘制条形图展示差异
x = np.arange(len(common_users))
width = 0.35

plt.bar(x - width/2, common_users['score_model1'], width, label='模型1分数')
plt.bar(x + width/2, common_users['score_model2'], width, label='模型2分数')

# 添加差异值标签
for i, diff in enumerate(common_users['difference']):
    plt.text(i, max(common_users.iloc[i, 0], common_users.iloc[i, 1]) + 0.01,
             f"差异: {diff:.4f}", ha='center')

plt.xlabel('用户ID')
plt.ylabel('分数')
plt.title('不同模型对同一用户的评分对比')
plt.xticks(x, common_users.index)
plt.legend()
plt.tight_layout()
plt.show()

# 计算并显示差异统计信息
print("\n差异统计信息:")
print(f"平均差异: {common_users['difference'].mean():.4f}")
print(f"差异标准差: {common_users['difference'].std():.4f}")
print(f"最大差异: {common_users['difference'].max():.4f} (用户: {common_users['difference'].idxmax()})")
print(f"最小差异: {common_users['difference'].min():.4f} (用户: {common_users['difference'].idxmin()})")
