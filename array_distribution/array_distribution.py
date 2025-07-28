import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 示例数据 - 可以替换为你的实际数组
data = [0.234, 0.23, 0.8, 1.0, 0.6, 0.45, 0.78, 0.92, 0.33, 0.55,
        0.12, 0.67, 0.89, 0.76, 0.44, 0.31, 0.59, 0.88, 0.91, 0.27,
        0.36, 0.72, 0.61, 0.53, 0.49, 0.83, 0.71, 0.65, 0.39, 0.51,0.51,0.51,0.51]

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 创建画布
plt.figure(figsize=(10, 6))

# 绘制直方图
n, bins, patches = plt.hist(data, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')

# 绘制核密度估计曲线
density = stats.gaussian_kde(data)
x = np.linspace(min(data), max(data), 200)
plt.plot(x, density(x), 'r-', linewidth=2)

# 添加标题和标签
plt.title('数据分布直方图与核密度估计', fontsize=15)
plt.xlabel('数值', fontsize=12)
plt.ylabel('密度', fontsize=12)

# 添加网格线
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 显示图形
plt.tight_layout()
plt.show()
