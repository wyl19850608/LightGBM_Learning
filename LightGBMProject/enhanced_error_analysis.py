import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 设置中文显示
# plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
# plt.rcParams['axes.unicode_minus'] = False

# ----------------------
# 1. 数据加载与准备
# ----------------------
def load_data(file_path):
    """加载预测结果数据并检查必要列"""
    try:
        df = pd.read_csv(file_path)
        required_cols = ['实际标签', '预测标签', '预测概率']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必要列: {missing_cols}")

        # 分类错误样本
        df['错误类型'] = '正确'
        df.loc[(df['实际标签'] == 0) & (df['预测标签'] == 1), '错误类型'] = '假阳性(FP)'  # 负样本中的错误
        df.loc[(df['实际标签'] == 1) & (df['预测标签'] == 0), '错误类型'] = '假阴性(FN)'  # 正样本中的错误

        print(f"数据加载完成，共 {len(df)} 条样本")
        print(f"正样本总数: {len(df[df['实际标签'] == 1])}")
        print(f"负样本总数: {len(df[df['实际标签'] == 0])}")
        print(f"假阴性(FN)样本数: {len(df[df['错误类型'] == '假阴性(FN)'])}")
        print(f"假阳性(FP)样本数: {len(df[df['错误类型'] == '假阳性(FP)'])}")

        return df
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        raise

# ----------------------
# 2. 单个指标的统计分析
# ----------------------
def analyze_single_indicator(df, indicator_col, top_n=5):
    """
    分析单个指标在正负样本错误中的统计差异
    :param df: 包含标签和错误类型的数据框
    :param indicator_col: 要分析的指标列名
    :param top_n: 类别型指标展示的 top N 众数
    """
    # 数据分组
    positive_samples = df[df['实际标签'] == 1]  # 正样本
    negative_samples = df[df['实际标签'] == 0]  # 负样本

    # 错误样本分组
    fn_samples = positive_samples[positive_samples['错误类型'] == '假阴性(FN)']  # 正样本中的错误
    fp_samples = negative_samples[negative_samples['错误类型'] == '假阳性(FP)']  # 负样本中的错误

    # 正确样本分组（用于对比）
    tp_samples = positive_samples[positive_samples['错误类型'] == '正确']  # 正样本中的正确
    tn_samples = negative_samples[negative_samples['错误类型'] == '正确']  # 负样本中的正确

    # 检查指标类型（数值型/类别型）
    is_numeric = pd.api.types.is_numeric_dtype(df[indicator_col])

    # 存储统计结果
    stats_results = {}

    # ----------------------
    # 数值型指标统计（均值、中位数等）
    # ----------------------
    if is_numeric:
        # 计算统计量
        stats_results['正样本-错误(FN)'] = {
            '均值': fn_samples[indicator_col].mean(),
            '中位数': fn_samples[indicator_col].median(),
            '标准差': fn_samples[indicator_col].std(),
            '最小值': fn_samples[indicator_col].min(),
            '最大值': fn_samples[indicator_col].max(),
            '样本数': len(fn_samples)
        }

        stats_results['正样本-正确(TP)'] = {
            '均值': tp_samples[indicator_col].mean(),
            '中位数': tp_samples[indicator_col].median(),
            '标准差': tp_samples[indicator_col].std(),
            '最小值': tp_samples[indicator_col].min(),
            '最大值': tp_samples[indicator_col].max(),
            '样本数': len(tp_samples)
        }

        stats_results['负样本-错误(FP)'] = {
            '均值': fp_samples[indicator_col].mean(),
            '中位数': fp_samples[indicator_col].median(),
            '标准差': fp_samples[indicator_col].std(),
            '最小值': fp_samples[indicator_col].min(),
            '最大值': fp_samples[indicator_col].max(),
            '样本数': len(fp_samples)
        }

        stats_results['负样本-正确(TN)'] = {
            '均值': tn_samples[indicator_col].mean(),
            '中位数': tn_samples[indicator_col].median(),
            '标准差': tn_samples[indicator_col].std(),
            '最小值': tn_samples[indicator_col].min(),
            '最大值': tn_samples[indicator_col].max(),
            '样本数': len(tn_samples)
        }

        # 转换为数据框便于展示
        stats_df = pd.DataFrame(stats_results).T

        # ----------------------
        # 打印统计结果
        # ----------------------
        print(f"\n===== 指标 '{indicator_col}' 统计分析（数值型） =====")
        print(stats_df.round(4))

        # 均值差异计算
        fn_tp_mean_diff = stats_results['正样本-错误(FN)']['均值'] - stats_results['正样本-正确(TP)']['均值']
        fp_tn_mean_diff = stats_results['负样本-错误(FP)']['均值'] - stats_results['负样本-正确(TN)']['均值']
        print(f"\n均值差异:")
        print(f"正样本错误(FN)与正确(TP)的均值差: {fn_tp_mean_diff:.4f}")
        print(f"负样本错误(FP)与正确(TN)的均值差: {fp_tn_mean_diff:.4f}")

        # ----------------------
        # 可视化差异
        # ----------------------
        # plt.figure(figsize=(12, 6))
        #
        # # 箱线图展示分布差异
        # plt.subplot(1, 2, 1)
        # plot_data = pd.concat([
        #     fn_samples[[indicator_col]].assign(group='正样本-错误(FN)'),
        #     tp_samples[[indicator_col]].assign(group='正样本-正确(TP)'),
        #     fp_samples[[indicator_col]].assign(group='负样本-错误(FP)'),
        #     tn_samples[[indicator_col]].assign(group='负样本-正确(TN)')
        # ])
        # sns.boxplot(x='group', y=indicator_col, data=plot_data)
        # plt.title(f'{indicator_col} 在各组样本中的分布')
        # plt.xticks(rotation=15)
        #
        # # 均值对比柱状图
        # plt.subplot(1, 2, 2)
        # mean_values = [
        #     stats_results['正样本-错误(FN)']['均值'],
        #     stats_results['正样本-正确(TP)']['均值'],
        #     stats_results['负样本-错误(FP)']['均值'],
        #     stats_results['负样本-正确(TN)']['均值']
        # ]
        # sns.barplot(x=['FN', 'TP', 'FP', 'TN'], y=mean_values)
        # plt.title(f'{indicator_col} 均值对比')
        # plt.ylabel('均值')
        #
        # plt.tight_layout()
        # plt.savefig(f'{indicator_col}_numeric_analysis.png', dpi=300)
        # plt.close()

        # 假设检验（判断均值差异是否显著）
        if len(fn_samples) >= 30 and len(tp_samples) >= 30:
            fn_tp_pvalue = stats.ttest_ind(fn_samples[indicator_col].dropna(),
                                           tp_samples[indicator_col].dropna()).pvalue
            print(f"正样本错误与正确的均值差异显著性（p值）: {fn_tp_pvalue:.4f} (p<0.05表示差异显著)")

        if len(fp_samples) >= 30 and len(tn_samples) >= 30:
            fp_tn_pvalue = stats.ttest_ind(fp_samples[indicator_col].dropna(),
                                           tn_samples[indicator_col].dropna()).pvalue
            print(f"负样本错误与正确的均值差异显著性（p值）: {fp_tn_pvalue:.4f} (p<0.05表示差异显著)")

    # ----------------------
    # 类别型指标统计（众数、频率等）
    # ----------------------
    else:
        # 计算众数和频率
        def get_mode_and_freq(data, col, top_n):
            if len(data) == 0:
                return pd.Series([None]*top_n, index=[f'众数{i+1}' for i in range(top_n)])
            mode_counts = data[col].value_counts()
            modes = mode_counts.index[:top_n].tolist()
            freqs = (mode_counts[:top_n] / len(data)).tolist()
            result = []
            for m, f in zip(modes, freqs):
                result.append(f"{m} ({f:.2%})")
            # 不足top_n的用空值填充
            while len(result) < top_n:
                result.append(None)
            return pd.Series(result, index=[f'众数{i+1}' for i in range(top_n)])

        # 统计各类样本的众数及频率
        fn_mode = get_mode_and_freq(fn_samples, indicator_col, top_n)
        tp_mode = get_mode_and_freq(tp_samples, indicator_col, top_n)
        fp_mode = get_mode_and_freq(fp_samples, indicator_col, top_n)
        tn_mode = get_mode_and_freq(tn_samples, indicator_col, top_n)

        # 组合结果
        stats_df = pd.DataFrame({
            '正样本-错误(FN)': fn_mode,
            '正样本-正确(TP)': tp_mode,
            '负样本-错误(FP)': fp_mode,
            '负样本-正确(TN)': tn_mode
        })

        # ----------------------
        # 打印统计结果
        # ----------------------
        print(f"\n===== 指标 '{indicator_col}' 统计分析（类别型） =====")
        print(stats_df)

        # ----------------------
        # 可视化差异（前3个主要类别）
        # plt.figure(figsize=(12, 8))
        # 选择每个组中最常见的3个类别
        top_categories = set()
        for s in [fn_mode, tp_mode, fp_mode, tn_mode]:
            for val in s.dropna():
                if val:
                    top_categories.add(val.split()[0])  # 提取类别名称
        top_categories = list(top_categories)[:3]  # 最多展示3个类别

        # 计算每个类别的频率
        plot_data = []
        for group_name, group_data in [
            ('正样本-错误(FN)', fn_samples),
            ('正样本-正确(TP)', tp_samples),
            ('负样本-错误(FP)', fp_samples),
            ('负样本-正确(TN)', tn_samples)
        ]:
            if len(group_data) == 0:
                continue
            total = len(group_data)
            for cat in top_categories:
                count = len(group_data[group_data[indicator_col].astype(str) == cat])
                plot_data.append({
                    '组别': group_name,
                    '类别': cat,
                    '频率': count / total
                })

        # if plot_data:
        #     plot_df = pd.DataFrame(plot_data)
        #     sns.barplot(x='组别', y='频率', hue='类别', data=plot_df)
        #     plt.title(f'{indicator_col} 主要类别的频率分布')
        #     plt.xticks(rotation=15)
        #     plt.ylabel('频率')
        #     plt.tight_layout()
        #     plt.savefig(f'{indicator_col}_categorical_analysis.png', dpi=300)
        #     plt.close()
        # else:
        #     print(f"指标 '{indicator_col}' 样本量不足，无法生成可视化图表")

    return stats_df

# ----------------------
# 3. 批量分析多个指标
# ----------------------
def analyze_multiple_indicators(df, exclude_cols=None, top_n=5):
    """批量分析多个指标"""
    if exclude_cols is None:
        exclude_cols = ['实际标签', '预测标签', '预测概率', '错误类型', 'cust_id']  # 默认排除标签和ID列

    # 获取所有待分析的指标列
    indicator_cols = [col for col in df.columns if col not in exclude_cols]
    print(f"\n共发现 {len(indicator_cols)} 个待分析指标")

    # 存储所有指标的分析结果
    all_results = {}

    # 逐个分析指标
    for col in indicator_cols:
        print(f"\n{'='*50}")
        stats_df = analyze_single_indicator(df, col, top_n)
        all_results[col] = stats_df

    return all_results

# ----------------------
# 主函数
# ----------------------
def main(file_path):
    # 加载数据
    df = load_data(file_path)

    # 批量分析指标（可自定义排除列）
    results = analyze_multiple_indicators(
        df,
        exclude_cols=['实际标签', '预测标签', '预测概率', '错误类型', 'cust_id'],
        top_n=3  # 类别型指标展示前3个众数
    )

    print("\n所有指标分析完成！")
    return results

if __name__ == "__main__":
    # 预测结果文件路径
    prediction_file = "test_predictions_with_features.csv"
    main(prediction_file)
