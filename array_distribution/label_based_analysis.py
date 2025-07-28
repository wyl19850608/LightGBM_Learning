import pandas as pd
import numpy as np

def calculate_group_statistics(group, group_name, data_name):
    """计算分组数据的统计指标"""
    scores = group['score']

    if len(scores) == 0:
        print(f"\n===== {data_name} 中 {group_name} 无数据 =====")
        return None

    # 计算基本统计量
    mean_val = scores.mean()
    median_val = scores.median()
    std_val = scores.std()

    # 计算分位数（25%、50%、75%、95%）
    quantiles = scores.quantile([0.25, 0.5, 0.75, 0.95])

    print(f"\n===== {data_name} 中 {group_name} 的统计指标 =====")
    print(f"样本数量: {len(scores)}")
    print(f"均值: {mean_val:.4f}")
    print(f"中位数: {median_val:.4f}")
    print(f"标准差: {std_val:.4f}")
    print("\n分位数:")
    for q, val in quantiles.items():
        print(f"  {int(q*100)}%: {val:.4f}")

    return scores

def calculate_group_difference_stats(merged, label, name1, name2):
    """计算特定label下两个文件共同用户的差异统计"""
    # 筛选特定label的样本（使用第一个文件的label，假设两个文件label一致）
    group = merged[merged[f'label_{name1}'] == label]

    if len(group) == 0:
        print(f"\n===== 无 label={label} 的共同用户 =====")
        return

    # 计算差异（第一个文件分数 - 第二个文件分数）
    diff_series = group[f'score_{name1}'] - group[f'score_{name2}']

    # 计算差异的统计量
    mean_diff = diff_series.mean()
    median_diff = diff_series.median()
    std_diff = diff_series.std()
    quantiles_diff = diff_series.quantile([0.25, 0.5, 0.75, 0.95])

    # 计算正负差异占比
    positive_ratio = (diff_series > 0).mean() * 100
    negative_ratio = (diff_series < 0).mean() * 100
    zero_ratio = (diff_series == 0).mean() * 100

    print(f"\n===== label={label} 时 {name1} 与 {name2} 的差异统计 =====")
    print(f"共同用户数量: {len(group)}")
    print(f"差异均值 ({name1} - {name2}): {mean_diff:.4f}")
    print(f"差异中位数: {median_diff:.4f}")
    print(f"差异标准差: {std_diff:.4f}")
    print("\n差异分位数:")
    for q, val in quantiles_diff.items():
        print(f"  {int(q*100)}%: {val:.4f}")
    print(f"\n正差异（{name1} > {name2}）占比: {positive_ratio:.2f}%")
    print(f"负差异（{name1} < {name2}）占比: {negative_ratio:.2f}%")
    print(f"差异为0占比: {zero_ratio:.2f}%")

# 读取CSV文件
# 请替换为你的实际文件路径
file1_path = 'file1.csv'
file2_path = 'file2.csv'

try:
    # 读取CSV文件
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # 检查必要的列是否存在
    required_columns = ['user_id', 'score', 'label']
    for df, name in [(df1, file1_path), (df2, file2_path)]:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"文件 {name} 缺少必要的列: {missing}")

    # 检查label是否只有0和1
    for df, name in [(df1, file1_path), (df2, file2_path)]:
        labels = df['label'].unique()
        invalid = [l for l in labels if l not in [0, 1]]
        if invalid:
            raise ValueError(f"文件 {name} 包含无效的label值: {invalid}，只能是0或1")

    # 按label分组计算每个文件的统计指标
    for label in [0, 1]:
        # 文件1的分组统计
        group1 = df1[df1['label'] == label]
        calculate_group_statistics(group1, f'label={label}', file1_path)

        # 文件2的分组统计
        group2 = df2[df2['label'] == label]
        calculate_group_statistics(group2, f'label={label}', file2_path)

    # 合并两个文件，获取共同用户
    merged = pd.merge(
        df1[['user_id', 'score', 'label']],
        df2[['user_id', 'score', 'label']],
        on='user_id',
        how='inner',
        suffixes=(f'_{file1_path}', f'_{file2_path}')
    )

    # 检查两个文件中同一用户的label是否一致
    label_mismatch = merged[merged[f'label_{file1_path}'] != merged[f'label_{file2_path}']]
    if len(label_mismatch) > 0:
        print(f"\n警告: 发现 {len(label_mismatch)} 个用户在两个文件中的label不一致")

    # 按label分组计算差异统计
    for label in [0, 1]:
        calculate_group_difference_stats(merged, label, file1_path, file2_path)

except FileNotFoundError as e:
    print(f"错误: 文件未找到 - {e}")
except Exception as e:
    print(f"处理过程中发生错误: {e}")
