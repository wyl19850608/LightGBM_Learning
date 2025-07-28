import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 设置中文显示
# plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
# plt.rcParams['axes.unicode_minus'] = False

# ----------------------
# 1. 加载数据
# ----------------------
def load_prediction_data(file_path):
    """加载预测结果数据"""
    try:
        df = pd.read_csv(file_path)
        print(f"成功加载数据，共 {len(df)} 条样本，{len(df.columns)} 个特征")

        # 检查必要列是否存在
        required_cols = ['实际标签', '预测标签', '预测概率']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必要列: {missing_cols}")

        return df
    except Exception as e:
        print(f"加载数据失败: {str(e)}")
        raise

# ----------------------
# 2. 错误样本分类
# ----------------------
def categorize_error_samples(df):
    """将样本分为TP、TN、FP、FN四类"""
    # 创建副本避免修改原始数据
    result_df = df.copy()

    # 分类错误类型
    result_df['错误类型'] = '正确'
    # 假阳性: 实际为0，预测为1
    result_df.loc[(result_df['实际标签'] == 0) & (result_df['预测标签'] == 1), '错误类型'] = '假阳性(FP)'
    # 假阴性: 实际为1，预测为0
    result_df.loc[(result_df['实际标签'] == 1) & (result_df['预测标签'] == 0), '错误类型'] = '假阴性(FN)'

    # 提取各类样本
    tp = result_df[(result_df['实际标签'] == 1) & (result_df['预测标签'] == 1)]
    tn = result_df[(result_df['实际标签'] == 0) & (result_df['预测标签'] == 0)]
    fp = result_df[result_df['错误类型'] == '假阳性(FP)']
    fn = result_df[result_df['错误类型'] == '假阴性(FN)']

    # 打印错误样本统计
    print(f"\n样本分类统计:")
    print(f"真阳性(TP): {len(tp)} 条 ({len(tp)/len(result_df):.2%})")
    print(f"真阴性(TN): {len(tn)} 条 ({len(tn)/len(result_df):.2%})")
    print(f"假阳性(FP): {len(fp)} 条 ({len(fp)/len(result_df):.2%})")
    print(f"假阴性(FN): {len(fn)} 条 ({len(fn)/len(result_df):.2%})")

    return result_df, tp, tn, fp, fn

# ----------------------
# 3. 错误样本指标计算
# ----------------------
def calculate_error_metrics(df):
    """计算错误相关指标"""
    y_true = df['实际标签']
    y_pred = df['预测标签']

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    if len(cm) == 1:
        # 处理只有一类的边缘情况
        tn, fp, fn, tp = 0, 0, 0, 0
        if y_true.iloc[0] == 0:
            tn = cm[0][0]
        else:
            tp = cm[0][0]
    else:
        tn, fp, fn, tp = cm.ravel()

    # 计算错误率指标
    metrics = {
        '总样本数': len(df),
        '准确率': (tp + tn) / len(df) if len(df) > 0 else 0,
        '错误率': (fp + fn) / len(df) if len(df) > 0 else 0,
        '假阳性率(FP率)': fp / (fp + tn) if (fp + tn) > 0 else 0,  # 在实际负例中被误判为正例的比例
        '假阴性率(FN率)': fn / (fn + tp) if (fn + tp) > 0 else 0,  # 在实际正例中被误判为负例的比例
        '假阳性数': fp,
        '假阴性数': fn
    }

    print("\n错误评估指标:")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")

    # 详细分类报告
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=['负例(0)', '正例(1)']))

    return metrics, cm

# ----------------------
# 4. 错误样本特征分析
# ----------------------
def analyze_error_features(tp, tn, fp, fn, top_n=10):
    """分析错误样本与正确样本的特征差异"""
    # 获取特征列（排除标签和预测相关列）
    feature_cols = [col for col in tp.columns if col not in ['实际标签', '预测标签', '预测概率', '错误类型']]
    if not feature_cols:
        print("没有可分析的特征列")
        return None

    # 选择数值型特征进行分析
    numeric_features = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(tp[col]):
            numeric_features.append(col)

    if not numeric_features:
        print("没有可分析的数值型特征")
        return None

    # 限制分析的特征数量，避免输出过多
    # analyze_features = numeric_features[:top_n]
    analyze_features = numeric_features
    print(f"\n分析前 {len(analyze_features)} 个数值特征在错误样本与正确样本中的差异...")

    # 创建结果保存目录
    import os
    if not os.path.exists('error_analysis_plots'):
        os.makedirs('error_analysis_plots')

    # 1. 计算特征均值差异
    feature_stats = pd.DataFrame()
    for feature in analyze_features:
        # 正例相关对比（TP vs FN）
        if len(tp) > 0 and len(fn) > 0:
            feature_stats.loc[feature, 'TP均值'] = tp[feature].mean()
            feature_stats.loc[feature, 'FN均值'] = fn[feature].mean()
            feature_stats.loc[feature, 'TP-FN差异(绝对值)'] = abs(tp[feature].mean() - fn[feature].mean())

        # 负例相关对比（TN vs FP）
        if len(tn) > 0 and len(fp) > 0:
            feature_stats.loc[feature, 'TN均值'] = tn[feature].mean()
            feature_stats.loc[feature, 'FP均值'] = fp[feature].mean()
            feature_stats.loc[feature, 'TN-FP差异(绝对值)'] = abs(tn[feature].mean() - fp[feature].mean())

    # 按差异排序并保存
    if not feature_stats.empty:
        feature_stats = feature_stats.sort_values('TP-FN差异(绝对值)', ascending=False)
        feature_stats.to_csv('error_analysis_plots/feature_mean_differences.csv')
        print("\n特征均值差异（TP vs FN:")
        print(feature_stats[['TP均值', 'FN均值', 'TP-FN差异(绝对值)']])

    # 2. 可视化特征分布差异
    # for feature in analyze_features[:5]:  # 只可视化前5个特征，避免过多文件
    #     # TP vs FN 分布对比
    #     if len(tp) > 0 and len(fn) > 0:
    #         plt.figure(figsize=(10, 6))
    #         sns.kdeplot(tp[feature], label='真阳性(TP)', fill=True, alpha=0.5)
    #         sns.kdeplot(fn[feature], label='假阴性(FN)', fill=True, alpha=0.5)
    #         plt.title(f'{feature} 在TP和FN中的分布对比')
    #         plt.xlabel(feature)
    #         plt.ylabel('密度')
    #         plt.legend()
    #         plt.tight_layout()
    #         plt.savefig(f'error_analysis_plots/{feature}_tp_vs_fn.png', dpi=300)
    #         plt.close()
    #
    #     # TN vs FP 分布对比
    #     if len(tn) > 0 and len(fp) > 0:
    #         plt.figure(figsize=(10, 6))
    #         sns.kdeplot(tn[feature], label='真阴性(TN)', fill=True, alpha=0.5)
    #         sns.kdeplot(fp[feature], label='假阳性(FP)', fill=True, alpha=0.5)
    #         plt.title(f'{feature} 在TN和FP中的分布对比')
    #         plt.xlabel(feature)
    #         plt.ylabel('密度')
    #         plt.legend()
    #         plt.tight_layout()
    #         plt.savefig(f'error_analysis_plots/{feature}_tn_vs_fp.png', dpi=300)
    #         plt.close()

    return feature_stats

# ----------------------
# 5. 可视化错误分析结果
# ----------------------
def visualize_error_results(cm, df):
    """可视化错误分析结果"""
    # 创建结果保存目录
    import os
    if not os.path.exists('error_analysis_plots'):
        os.makedirs('error_analysis_plots')

    # 1. 混淆矩阵可视化
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
    #             xticklabels=['预测负例', '预测正例'],
    #             yticklabels=['实际负例', '实际正例'])
    # plt.xlabel('预测标签')
    # plt.ylabel('实际标签')
    # plt.title('混淆矩阵')
    # plt.tight_layout()
    # plt.savefig('error_analysis_plots/confusion_matrix.png', dpi=300)
    # plt.close()

    # 2. 预测概率分布对比（错误样本vs正确样本）
    # plt.figure(figsize=(10, 6))
    # sns.histplot(df[df['错误类型'] == '正确']['预测概率'],
    #              label='正确样本', bins=20, alpha=0.5)
    # sns.histplot(df[df['错误类型'] != '正确']['预测概率'],
    #              label='错误样本', bins=20, alpha=0.5)
    # plt.axvline(x=0.5, color='r', linestyle='--', label='阈值(0.5)')
    # plt.xlabel('预测概率')
    # plt.ylabel('样本数')
    # plt.title('正确样本与错误样本的预测概率分布')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('error_analysis_plots/probability_distribution.png', dpi=300)
    # plt.close()
    #
    # # 3. 错误类型占比
    # error_counts = df['错误类型'].value_counts()
    # plt.figure(figsize=(8, 6))
    # error_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90)
    # plt.title('错误类型占比')
    # plt.ylabel('')  # 去除y轴标签
    # plt.tight_layout()
    # plt.savefig('error_analysis_plots/error_type_pie.png', dpi=300)
    # plt.close()

    print("\n错误分析可视化完成，图表已保存到 error_analysis_plots 目录")

# ----------------------
# 6. 保存错误样本数据
# ----------------------
def save_error_samples(df):
    """保存错误样本数据供进一步分析"""
    # 创建结果保存目录
    import os
    if not os.path.exists('error_samples'):
        os.makedirs('error_samples')

    # 保存所有错误样本
    error_samples = df[df['错误类型'] != '正确']
    error_samples.to_csv('error_samples/all_errors.csv', index=False)

    # 分别保存假阳性和假阴性样本
    fp_samples = df[df['错误类型'] == '假阳性(FP)']
    fn_samples = df[df['错误类型'] == '假阴性(FN)']

    fp_samples.to_csv('error_samples/false_positives.csv', index=False)
    fn_samples.to_csv('error_samples/false_negatives.csv', index=False)

    print(f"\n错误样本已保存到 error_samples 目录:")
    print(f"- 所有错误样本: {len(error_samples)} 条")
    print(f"- 假阳性样本: {len(fp_samples)} 条")
    print(f"- 假阴性样本: {len(fn_samples)} 条")

# ----------------------
# 主函数
# ----------------------
def main(prediction_file_path):
    # 1. 加载预测结果数据
    df = load_prediction_data(prediction_file_path)

    # 2. 分类错误样本
    result_df, tp, tn, fp, fn = categorize_error_samples(df)

    # 3. 计算错误指标
    metrics, cm = calculate_error_metrics(result_df)

    # 4. 分析错误样本特征
    feature_stats = analyze_error_features(tp, tn, fp, fn)

    # 5. 可视化错误分析结果
    visualize_error_results(cm, result_df)

    # 6. 保存错误样本数据
    save_error_samples(result_df)

    print("\n错误样本评估分析完成!")
    return result_df, metrics

if __name__ == "__main__":
    # 预测结果文件路径
    prediction_file = "test_predictions_with_features.csv"
    main(prediction_file)
