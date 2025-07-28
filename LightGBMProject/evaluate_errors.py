import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import shap
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False

# ----------------------
# 1. 数据加载
# ----------------------
# 在load_data_and_model函数中添加缺失值检查和处理
def load_data_and_model(model_path, test_data_path, prediction_path):
    """加载模型、测试数据和预测结果"""
    # 加载模型
    model = lgb.Booster(model_file=model_path)

    # 加载测试数据（包含特征和真实标签）
    test_data = pd.read_csv(test_data_path)

    # 加载预测结果
    predictions = pd.read_csv(prediction_path)

    # 合并数据
    if 'Unnamed: 0' in predictions.columns:
        predictions = predictions.drop(columns=['Unnamed: 0'])

    # 确保索引一致
    test_data = test_data.reset_index(drop=True)
    predictions = predictions.reset_index(drop=True)
    combined = pd.concat([test_data, predictions], axis=1)

    # 确保必要列存在
    required_cols = ['实际标签', '预测标签', '预测概率']
    for col in required_cols:
        if col not in combined.columns:
            raise ValueError(f"预测结果文件缺少必要列: {col}")

    # 新增：检查并处理实际标签中的缺失值
    if combined['实际标签'].isnull().any():
        nan_count = combined['实际标签'].isnull().sum()
        print(f"警告：实际标签中存在{nan_count}个缺失值，将被删除")
        combined = combined.dropna(subset=['实际标签'])
        # 重置索引
        combined = combined.reset_index(drop=True)

    return model, combined

# ----------------------
# 2. 错误样本分类
# ----------------------
def categorize_errors(combined_data):
    """将样本分为TP、TN、FP、FN四类"""
    # 创建错误类型列
    combined_data['错误类型'] = '正确'

    # 假阳性: 实际为0，预测为1
    combined_data.loc[
        (combined_data['实际标签'] == 0) & (combined_data['预测标签'] == 1),
        '错误类型'
    ] = '假阳性(FP)'

    # 假阴性: 实际为1，预测为0
    combined_data.loc[
        (combined_data['实际标签'] == 1) & (combined_data['预测标签'] == 0),
        '错误类型'
    ] = '假阴性(FN)'

    # 提取各类样本
    tp = combined_data[(combined_data['实际标签'] == 1) & (combined_data['预测标签'] == 1)]
    tn = combined_data[(combined_data['实际标签'] == 0) & (combined_data['预测标签'] == 0)]
    fp = combined_data[combined_data['错误类型'] == '假阳性(FP)']
    fn = combined_data[combined_data['错误类型'] == '假阴性(FN)']

    # 打印错误样本比例
    print(f"总样本数: {len(combined_data)}")
    print(f"真阳性(TP): {len(tp)} ({len(tp)/len(combined_data):.2%})")
    print(f"真阴性(TN): {len(tn)} ({len(tn)/len(combined_data):.2%})")
    print(f"假阳性(FP): {len(fp)} ({len(fp)/len(combined_data):.2%})")
    print(f"假阴性(FN): {len(fn)} ({len(fn)/len(combined_data):.2%})")

    # 打印详细分类报告
    print("\n分类报告:")
    print(classification_report(
        combined_data['实际标签'],
        combined_data['预测标签'],
        target_names=['负例(0)', '正例(1)']
    ))

    return combined_data, tp, tn, fp, fn

# ----------------------
# 3. 错误样本特征分布分析
# ----------------------
def analyze_feature_distributions(all_data, tp, tn, fp, fn, top_features=None, n_features=10):
    """分析关键特征在错误样本与正确样本中的分布差异"""
    # 获取模型特征（排除标签和预测相关列）
    exclude_cols = ['实际标签', '预测标签', '预测概率', '错误类型', 'cust_recv_time', 'touch_time']
    feature_cols = [col for col in all_data.columns if col not in exclude_cols]

    # 如果未指定top_features，使用所有特征
    if top_features is None:
        top_features = feature_cols[:n_features]

    # 创建结果保存目录
    import os
    if not os.path.exists('error_analysis_plots'):
        os.makedirs('error_analysis_plots')

    # 1. 数值特征分布对比
    for feature in top_features:
        # 检查是否为数值特征
        if pd.api.types.is_numeric_dtype(all_data[feature]):
            plt.figure(figsize=(12, 6))

            # 绘制核密度图对比分布
            sns.kdeplot(tp[feature], label='真阳性(TP)', fill=True, alpha=0.5)
            sns.kdeplot(fn[feature], label='假阴性(FN)', fill=True, alpha=0.5)
            plt.title(f'{feature} 在TP和FN中的分布对比')
            plt.xlabel(feature)
            plt.ylabel('密度')
            plt.legend()
            plt.savefig(f'error_analysis_plots/{feature}_tp_vs_fn.png', dpi=300)
            plt.close()

            plt.figure(figsize=(12, 6))
            sns.kdeplot(tn[feature], label='真阴性(TN)', fill=True, alpha=0.5)
            sns.kdeplot(fp[feature], label='假阳性(FP)', fill=True, alpha=0.5)
            plt.title(f'{feature} 在TN和FP中的分布对比')
            plt.xlabel(feature)
            plt.ylabel('密度')
            plt.legend()
            plt.savefig(f'error_analysis_plots/{feature}_tn_vs_fp.png', dpi=300)
            plt.close()

    # 2. 计算特征均值差异
    feature_stats = pd.DataFrame()
    for feature in feature_cols:
        if pd.api.types.is_numeric_dtype(all_data[feature]):
            feature_stats.loc[feature, 'TP均值'] = tp[feature].mean()
            feature_stats.loc[feature, 'FN均值'] = fn[feature].mean()
            feature_stats.loc[feature, 'TP-FN差异'] = abs(tp[feature].mean() - fn[feature].mean())

            feature_stats.loc[feature, 'TN均值'] = tn[feature].mean()
            feature_stats.loc[feature, 'FP均值'] = fp[feature].mean()
            feature_stats.loc[feature, 'TN-FP差异'] = abs(tn[feature].mean() - fp[feature].mean())

    # 按差异排序并保存
    feature_stats = feature_stats.sort_values('TP-FN差异', ascending=False)
    feature_stats.to_csv('error_analysis_plots/feature_mean_differences.csv')

    print("\n特征均值差异最大的前10个特征（TP vs FN）:")
    print(feature_stats[['TP均值', 'FN均值', 'TP-FN差异']].head(10))

    return feature_stats

# ----------------------
# 4. 阈值分析与优化
# ----------------------
def analyze_threshold(all_data):
    """分析不同阈值对错误样本数量的影响"""
    # 计算不同阈值下的精确率和召回率
    precision, recall, thresholds = precision_recall_curve(
        all_data['实际标签'],
        all_data['预测概率']
    )

    # 绘制PR曲线
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, marker='.', label='PR曲线')
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('精确率-召回率曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig('error_analysis_plots/precision_recall_curve.png', dpi=300)
    plt.close()

    # 计算不同阈值下的错误样本数量
    threshold_analysis = []
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        pred_label = (all_data['预测概率'] >= threshold).astype(int)
        cm = confusion_matrix(all_data['实际标签'], pred_label)
        if len(cm) < 2:  # 处理边缘情况
            continue
        tn, fp, fn, tp = cm.ravel()
        threshold_analysis.append({
            '阈值': threshold,
            '假阳性(FP)': fp,
            '假阴性(FN)': fn,
            '总错误数': fp + fn,
            '精确率': tp / (tp + fp) if (tp + fp) > 0 else 0,
            '召回率': tp / (tp + fn) if (tp + fn) > 0 else 0
        })

    threshold_df = pd.DataFrame(threshold_analysis)
    threshold_df.to_csv('error_analysis_plots/threshold_analysis.csv', index=False)

    print("\n不同阈值下的错误分析:")
    print(threshold_df)

    return threshold_df

# ----------------------
# 5. SHAP值分析错误原因
# ----------------------
def shap_analysis(model, all_data, fp, fn, top_n=10):
    """使用SHAP值分析导致错误预测的关键特征"""
    # 提取特征列
    exclude_cols = ['实际标签', '预测标签', '预测概率', '错误类型', 'cust_recv_time', 'touch_time']
    feature_cols = [col for col in all_data.columns if col not in exclude_cols]

    # 创建SHAP解释器
    explainer = shap.TreeExplainer(model)

    # 计算假阳性样本的SHAP值
    if len(fp) > 0:
        fp_features = fp[feature_cols]
        fp_shap_values = explainer.shap_values(fp_features)

        # 绘制假阳性样本的SHAP摘要图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(fp_shap_values, fp_features, feature_names=feature_cols, plot_size=(12, 8))
        plt.title('假阳性(FP)样本的SHAP值摘要')
        plt.tight_layout()
        plt.savefig('error_analysis_plots/shap_fp_summary.png', dpi=300)
        plt.close()

    # 计算假阴性样本的SHAP值
    if len(fn) > 0:
        fn_features = fn[feature_cols]
        fn_shap_values = explainer.shap_values(fn_features)

        # 绘制假阴性样本的SHAP摘要图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(fn_shap_values, fn_features, feature_names=feature_cols, plot_size=(12, 8))
        plt.title('假阴性(FN)样本的SHAP值摘要')
        plt.tight_layout()
        plt.savefig('error_analysis_plots/shap_fn_summary.png', dpi=300)
        plt.close()

    print("\nSHAP值分析完成，已生成假阳性和假阴性样本的特征重要性图")

# ----------------------
# 6. 错误样本的共同特征分析
# ----------------------
def analyze_common_patterns(fp, fn, top_n=5):
    """分析错误样本中最常见的特征模式"""
    # 分析假阳性样本的共同特征
    fp_patterns = {}
    if len(fp) > 0:
        # 对分类特征统计最常见值
        cat_cols = fp.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            top_values = fp[col].value_counts().head(top_n)
            fp_patterns[col] = dict(top_values)

        # 对数值特征统计分布区间
        num_cols = fp.select_dtypes(include=['int64', 'float64']).columns
        for col in num_cols:
            # 计算分位数
            q25, q50, q75 = fp[col].quantile([0.25, 0.5, 0.75])
            fp_patterns[col] = f"中位数: {q50:.2f}, 四分位: [{q25:.2f}, {q75:.2f}]"

    # 分析假阴性样本的共同特征
    fn_patterns = {}
    if len(fn) > 0:
        # 对分类特征统计最常见值
        cat_cols = fn.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            top_values = fn[col].value_counts().head(top_n)
            fn_patterns[col] = dict(top_values)

        # 对数值特征统计分布区间
        num_cols = fn.select_dtypes(include=['int64', 'float64']).columns
        for col in num_cols:
            # 计算分位数
            q25, q50, q75 = fn[col].quantile([0.25, 0.5, 0.75])
            fn_patterns[col] = f"中位数: {q50:.2f}, 四分位: [{q25:.2f}, {q75:.2f}]"

    # 保存分析结果
    with open('error_analysis_plots/fp_common_patterns.txt', 'w', encoding='utf-8') as f:
        f.write("假阳性(FP)样本的常见特征模式:\n")
        for key, value in fp_patterns.items():
            f.write(f"{key}: {value}\n")

    with open('error_analysis_plots/fn_common_patterns.txt', 'w', encoding='utf-8') as f:
        f.write("假阴性(FN)样本的常见特征模式:\n")
        for key, value in fn_patterns.items():
            f.write(f"{key}: {value}\n")

    print("\n错误样本的常见特征模式已保存")
    return fp_patterns, fn_patterns

# ----------------------
# 主函数
# ----------------------
def main(model_path, test_data_path, prediction_path):
    # 1. 加载数据和模型
    model, combined_data = load_data_and_model(model_path, test_data_path, prediction_path)

    # 2. 分类错误样本
    combined_data, tp, tn, fp, fn = categorize_errors(combined_data)

    # 3. 分析特征分布差异
    feature_stats = analyze_feature_distributions(combined_data, tp, tn, fp, fn)

    # 4. 阈值分析
    threshold_df = analyze_threshold(combined_data)

    # 5. SHAP值分析
    shap_analysis(model, combined_data, fp, fn)

    # 6. 常见模式分析
    fp_patterns, fn_patterns = analyze_common_patterns(fp, fn)

    print("\n错误样本分析完成！所有结果已保存到 error_analysis_plots 目录")
    print("关键发现建议:")
    if len(fp) > 0:
        print(f"- 假阳性样本主要集中在 {list(fp_patterns.keys())[:3]} 等特征的特定区间")
    if len(fn) > 0:
        print(f"- 假阴性样本主要集中在 {list(fn_patterns.keys())[:3]} 等特征的特定区间")
    print(f"- 建议尝试阈值 {threshold_df.loc[threshold_df['总错误数'].idxmin()]['阈值']} 以减少总错误数")

if __name__ == "__main__":
    # 配置文件路径
    MODEL_PATH = "credit_approval_model.txt"          # 模型文件路径
    TEST_DATA_PATH = "simulated_credit_data.csv"      # 测试数据路径（包含特征和真实标签）
    PREDICTION_PATH = "test_predictions.csv"          # 预测结果路径

    main(MODEL_PATH, TEST_DATA_PATH, PREDICTION_PATH)
