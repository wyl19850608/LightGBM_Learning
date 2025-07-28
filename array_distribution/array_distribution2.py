import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 第一组数据
data1 = [0.234, 0.23, 0.8, 1.0, 0.6, 0.45, 0.78, 0.92, 0.33, 0.55,
         0.12, 0.67, 0.89, 0.76, 0.44, 0.31, 0.59, 0.88, 0.91, 0.27]

# 第二组数据
data2 = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 1.0,
         0.15, 0.35, 0.55, 0.75, 0.95, 0.25, 0.45, 0.65, 0.85, 0.05]

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 创建画布
plt.figure(figsize=(12, 7))

# 绘制第一组数据的直方图
n1, bins1, patches1 = plt.hist(data1, bins=50, density=True, alpha=0.5,
                               color='skyblue', edgecolor='black', label='第一组数据')

# 绘制第一组数据的核密度估计曲线
density1 = stats.gaussian_kde(data1)
x1 = np.linspace(min(data1), max(data1), 200)
plt.plot(x1, density1(x1), 'b-', linewidth=2)

# 绘制第二组数据的直方图
n2, bins2, patches2 = plt.hist(data2, bins=50, density=True, alpha=0.5,
                               color='lightgreen', edgecolor='black', label='第二组数据')

# 绘制第二组数据的核密度估计曲线
density2 = stats.gaussian_kde(data2)
x2 = np.linspace(min(data2), max(data2), 200)
plt.plot(x2, density2(x2), 'g-', linewidth=2)

# 添加总标题和坐标轴标签
plt.title('两组数据的分布对比', fontsize=16)
plt.xlabel('数值', fontsize=12)
plt.ylabel('密度', fontsize=12)

# 添加图例
plt.legend(fontsize=12)

# 添加网格线
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 显示图形
plt.tight_layout()
plt.show()
