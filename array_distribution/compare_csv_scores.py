import pandas as pd
import numpy as np

def calculate_statistics(df, name):
    """计算单个DataFrame中score列的统计指标"""
    scores = df['score']

    # 计算基本统计量
    mean_val = scores.mean()
    median_val = scores.median()
    std_val = scores.std()

    # 计算分位数（25%、50%、75%、95%）
    quantiles = scores.quantile([0.25, 0.5, 0.75, 0.95])

    print(f"\n===== {name} 统计指标 =====")
    print(f"均值: {mean_val:.4f}")
    print(f"中位数: {median_val:.4f}")
    print(f"标准差: {std_val:.4f}")
    print("\n分位数:")
    for q, val in quantiles.items():
        print(f"  {int(q*100)}%: {val:.4f}")

    return scores

def calculate_difference_stats(df1, df2, name1, name2):
    """计算两个DataFrame中共同用户的分数差异统计"""
    # 按user_id合并，只保留共同用户
    merged = pd.merge(
        df1[['user_id', 'score']],
        df2[['user_id', 'score']],
        on='user_id',
        how='inner',
        suffixes=(f'_{name1}', f'_{name2}')
    )

    # 计算差异（第一个文件分数 - 第二个文件分数）
    merged['difference'] = merged[f'score_{name1}'] - merged[f'score_{name2}']
    diff_series = merged['difference']

    # 计算差异的统计量
    mean_diff = diff_series.mean()
    median_diff = diff_series.median()
    std_diff = diff_series.std()
    quantiles_diff = diff_series.quantile([0.25, 0.5, 0.75, 0.95])

    # 计算正负差异占比
    positive_ratio = (diff_series > 0).mean() * 100
    negative_ratio = (diff_series < 0).mean() * 100
    zero_ratio = (diff_series == 0).mean() * 100

    print(f"\n===== {name1} 与 {name2} 分数差异统计 =====")
    print(f"差异均值 ({name1} - {name2}): {mean_diff:.4f}")
    print(f"差异中位数: {median_diff:.4f}")
    print(f"差异标准差: {std_diff:.4f}")
    print("\n差异分位数:")
    for q, val in quantiles_diff.items():
        print(f"  {int(q*100)}%: {val:.4f}")
    print(f"\n正差异（{name1} > {name2}）占比: {positive_ratio:.2f}%")
    print(f"负差异（{name1} < {name2}）占比: {negative_ratio:.2f}%")
    print(f"差异为0占比: {zero_ratio:.2f}%")
    print(f"\n共同用户数量: {len(merged)}")

# 读取CSV文件
# 请替换为你的实际文件路径
file1_path = 'file1.csv'
file2_path = 'file2.csv'

try:
    # 读取CSV文件
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # 检查必要的列是否存在
    required_columns = ['user_id', 'score']
    for df, name in [(df1, file1_path), (df2, file2_path)]:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"文件 {name} 缺少必要的列: {missing}")

    # 计算每个文件的基本统计指标
    scores1 = calculate_statistics(df1, file1_path)
    scores2 = calculate_statistics(df2, file2_path)

    # 计算两个文件共同用户的差异统计
    calculate_difference_stats(df1, df2, file1_path, file2_path)

except FileNotFoundError as e:
    print(f"错误: 文件未找到 - {e}")
except Exception as e:
    print(f"处理过程中发生错误: {e}")
