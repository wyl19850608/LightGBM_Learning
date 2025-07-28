import pandas as pd
import numpy as np
import lightgbm as lgb
# import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve
)
import shap
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
# plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
# plt.rcParams['axes.unicode_minus'] = False

# ----------------------
# 1. 数据加载与预处理
# ----------------------
def load_data_and_model(model_path, test_data_path, prediction_path, feature_names_path=None):
    """加载模型、测试数据和预测结果，并进行预处理"""
    # 加载模型
    model = lgb.Booster(model_file=model_path)

    # 获取训练时使用的特征名称
    train_feature_names = model.feature_name()
    print(f"训练模型使用了 {len(train_feature_names)} 个特征")

    # 加载测试数据（包含特征和真实标签）
    test_data = pd.read_csv(test_data_path)

    # 检查并对齐特征
    missing_features = [f for f in train_feature_names if f not in test_data.columns]
    extra_features = [f for f in test_data.columns if f not in train_feature_names and f not in ['实际标签']]

    if missing_features:
        print(f"警告：测试数据缺少 {len(missing_features)} 个训练时的特征，将添加这些特征并填充为0")
        for f in missing_features:
            test_data[f] = 0  # 用0填充缺失的特征

    if extra_features:
        print(f"警告：测试数据包含 {len(extra_features)} 个训练时没有的特征，将删除这些特征")
        test_data = test_data.drop(columns=extra_features)

    # 确保特征顺序与训练时一致
    # 安全处理：检查'实际标签'是否存在
    if '实际标签' in test_data.columns:
        test_data = test_data.reindex(columns=train_feature_names + ['实际标签'], fill_value=0)
    else:
        test_data = test_data.reindex(columns=train_feature_names, fill_value=0)
        print("警告：测试数据中未找到'实际标签'列")

    # 加载预测结果
    predictions = pd.read_csv(prediction_path)

    # 合并数据
    if 'Unnamed: 0' in predictions.columns:
        predictions = predictions.drop(columns=['Unnamed: 0'])

    # 确保索引一致
    test_data = test_data.reset_index(drop=True)
    predictions = predictions.reset_index(drop=True)
    combined = pd.concat([test_data, predictions], axis=1)

    print("combined.columns:",combined.columns)
    # 确保必要列存在
    required_cols = ['实际标签', '预测标签', '预测概率']
    for col in required_cols:
        if col not in combined.columns:
            raise ValueError(f"预测结果文件缺少必要列: {col}")

    # 处理实际标签中的缺失值 - 修复此处的错误
    # 使用any()明确检查是否有缺失值
    if combined['实际标签'].isnull().any():
        nan_count = combined['实际标签'].isnull().sum()
        print(f"警告：实际标签中存在{nan_count}个缺失值，将被删除")
        combined = combined.dropna(subset=['实际标签'])
        combined = combined.reset_index(drop=True)

    return model, combined, train_feature_names

# ----------------------
# 2. 特征类型转换
# ----------------------
def convert_categorical_features(data, feature_names):
    """将object类型的分类特征转换为数值类型"""
    # 只处理模型使用的特征
    features_to_process = [f for f in feature_names if f in data.columns]

    # 识别所有object类型的列
    object_cols = [col for col in features_to_process
                   if data[col].dtype == 'object']

    if object_cols:
        print(f"\n发现{len(object_cols)}个object类型特征，将转换为数值类型:")
        print(", ".join(object_cols))

        # 创建一个副本避免修改原始数据
        data_converted = data.copy()

        # 对于每个object类型列，使用映射转换为数值
        for col in object_cols:
            # 检查是否是布尔型特征（只包含两个值）
            unique_vals = data_converted[col].unique()
            if len(unique_vals) <= 2:
                # 对于二值特征，映射为0和1
                mapping = {val: i for i, val in enumerate(unique_vals)}
                data_converted[col] = data_converted[col].map(mapping)
                print(f"  {col}: {mapping}")
            else:
                # 对于多类别特征，使用标签编码
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                data_converted[col] = le.fit_transform(data_converted[col])
                print(f"  {col}: 已使用LabelEncoder转换，共{len(le.classes_)}个类别")

        return data_converted
    else:
        return data

# ----------------------
# 3. 错误样本分类与指标评估
# ----------------------
def categorize_errors(combined_data):
    """将样本分为TP、TN、FP、FN四类并计算详细评估指标"""
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

    # 计算详细评估指标
    y_true = combined_data['实际标签']
    y_pred = combined_data['预测标签']
    y_prob = combined_data['预测概率']

    # 单独计算各指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0

    # 打印详细指标
    print("\n===== 详细评估指标 =====")
    print(f"准确率(Accuracy): {accuracy:.4f} - 所有样本中预测正确的比例")
    print(f"精确率(Precision): {precision:.4f} - 预测为正例的样本中实际为正例的比例")
    print(f"召回率(Recall): {recall:.4f} - 实际为正例的样本中被正确预测的比例")
    print(f"F1分数(F1-Score): {f1:.4f} - 精确率和召回率的调和平均")
    print(f"AUC-ROC: {auc:.4f} - 模型区分正负例的能力")
    print("\n===== 分类报告 =====")
    print(classification_report(
        y_true, y_pred, target_names=['负例(0)', '正例(1)']
    ))

    # 保存指标到文件
    metrics_df = pd.DataFrame({
        '指标': ['准确率(Accuracy)', '精确率(Precision)', '召回率(Recall)', 'F1分数(F1-Score)', 'AUC-ROC'],
        '值': [accuracy, precision, recall, f1, auc],
        '说明': [
            '所有样本中预测正确的比例',
            '预测为正例的样本中实际为正例的比例',
            '实际为正例的样本中被正确预测的比例',
            '精确率和召回率的调和平均',
            '模型区分正负例的能力'
        ]
    })
    metrics_df.to_csv('error_analysis_plots/evaluation_metrics.csv', index=False)

    return combined_data, tp, tn, fp, fn

# ----------------------
# 4. 错误样本特征分布分析
# ----------------------
def analyze_feature_distributions(all_data, tp, tn, fp, fn, train_feature_names, top_features=None, n_features=10):
    """分析关键特征在错误样本与正确样本中的分布差异"""
    # 使用训练时的特征列表
    feature_cols = train_feature_names

    # 如果未指定top_features，使用所有特征
    if top_features is None:
        top_features = feature_cols[:n_features]

    # 创建结果保存目录
    import os
    if not os.path.exists('error_analysis_plots'):
        os.makedirs('error_analysis_plots')

    # 1. 数值特征分布对比
    # for feature in top_features:
    #     # 检查是否为数值特征
    #     if pd.api.types.is_numeric_dtype(all_data[feature]):
    #         plt.figure(figsize=(12, 6))
    #
    #         # 绘制核密度图对比分布
    #         sns.kdeplot(tp[feature], label='真阳性(TP)', fill=True, alpha=0.5)
    #         sns.kdeplot(fn[feature], label='假阴性(FN)', fill=True, alpha=0.5)
    #         plt.title(f'{feature} 在TP和FN中的分布对比')
    #         plt.xlabel(feature)
    #         plt.ylabel('密度')
    #         plt.legend()
    #         plt.savefig(f'error_analysis_plots/{feature}_tp_vs_fn.png', dpi=300)
    #         plt.close()
    #
    #         plt.figure(figsize=(12, 6))
    #         sns.kdeplot(tn[feature], label='真阴性(TN)', fill=True, alpha=0.5)
    #         sns.kdeplot(fp[feature], label='假阳性(FP)', fill=True, alpha=0.5)
    #         plt.title(f'{feature} 在TN和FP中的分布对比')
    #         plt.xlabel(feature)
    #         plt.ylabel('密度')
    #         plt.legend()
    #         plt.savefig(f'error_analysis_plots/{feature}_tn_vs_fp.png', dpi=300)
    #         plt.close()

    # 2. 计算特征均值差异
    feature_stats = pd.DataFrame()
    for feature in feature_cols:
        if pd.api.types.is_numeric_dtype(all_data[feature]):
            feature_stats.loc[feature, 'TP均值'] = tp[feature].mean() if len(tp) > 0 else np.nan
            feature_stats.loc[feature, 'FN均值'] = fn[feature].mean() if len(fn) > 0 else np.nan
            feature_stats.loc[feature, 'TP-FN差异'] = abs(feature_stats.loc[feature, 'TP均值'] - feature_stats.loc[feature, 'FN均值'])

            feature_stats.loc[feature, 'TN均值'] = tn[feature].mean() if len(tn) > 0 else np.nan
            feature_stats.loc[feature, 'FP均值'] = fp[feature].mean() if len(fp) > 0 else np.nan
            feature_stats.loc[feature, 'TN-FP差异'] = abs(feature_stats.loc[feature, 'TN均值'] - feature_stats.loc[feature, 'FP均值'])

    # 按差异排序并保存
    feature_stats = feature_stats.sort_values('TP-FN差异', ascending=False)
    feature_stats.to_csv('error_analysis_plots/feature_mean_differences.csv')

    print("\n特征均值差异最大的前10个特征（TP vs FN）:")
    print(feature_stats[['TP均值', 'FN均值', 'TP-FN差异']].head(10))

    return feature_stats

# ----------------------
# 5. 阈值分析与优化
# ----------------------
def analyze_threshold(all_data):
    """分析不同阈值对错误样本数量和指标的影响"""
    y_true = all_data['实际标签']
    y_prob = all_data['预测概率']

    # 计算不同阈值下的精确率和召回率
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # 绘制PR曲线
    # plt.figure(figsize=(10, 6))
    # plt.plot(recall, precision, marker='.', label='PR曲线')
    # plt.xlabel('召回率')
    # plt.ylabel('精确率')
    # plt.title('精确率-召回率曲线')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('error_analysis_plots/precision_recall_curve.png', dpi=300)
    # plt.close()
    #
    # # 绘制ROC曲线
    # fpr, tpr, _ = roc_curve(y_true, y_prob)
    # plt.figure(figsize=(10, 6))
    # plt.plot(fpr, tpr, marker='.', label=f'ROC曲线 (AUC = {roc_auc_score(y_true, y_prob):.4f})')
    # plt.plot([0, 1], [0, 1], 'k--')  # 随机猜测的基准线
    # plt.xlabel('假正例率(FPR)')
    # plt.ylabel('真正例率(TPR)')
    # plt.title('ROC曲线')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('error_analysis_plots/roc_curve.png', dpi=300)
    # plt.close()

    # 计算不同阈值下的错误样本数量和指标
    threshold_analysis = []
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        pred_label = (all_data['预测概率'] >= threshold).astype(int)
        cm = confusion_matrix(y_true, pred_label)
        if len(cm) < 2:  # 处理边缘情况
            continue
        tn, fp, fn, tp = cm.ravel()

        # 计算各阈值下的指标
        acc = accuracy_score(y_true, pred_label)
        prec = precision_score(y_true, pred_label) if (tp + fp) > 0 else 0
        rec = recall_score(y_true, pred_label) if (tp + fn) > 0 else 0
        f1 = f1_score(y_true, pred_label) if (prec + rec) > 0 else 0

        threshold_analysis.append({
            '阈值': threshold,
            '假阳性(FP)': fp,
            '假阴性(FN)': fn,
            '总错误数': fp + fn,
            '准确率': acc,
            '精确率': prec,
            '召回率': rec,
            'F1分数': f1
        })

    threshold_df = pd.DataFrame(threshold_analysis)
    threshold_df.to_csv('error_analysis_plots/threshold_analysis.csv', index=False)

    print("\n不同阈值下的错误与指标分析:")
    print(threshold_df.round(4))  # 保留4位小数

    return threshold_df

# ----------------------
# 6. SHAP值分析错误原因
# ----------------------
def shap_analysis(model, all_data, fp, fn, train_feature_names, top_n=10):
    """使用SHAP值分析导致错误预测的关键特征"""
    # 使用训练时的特征列表
    feature_cols = train_feature_names

    # 创建SHAP解释器
    explainer = shap.TreeExplainer(model)

    # 计算假阳性样本的SHAP值
    if len(fp) > 0:
        # 只选择训练时使用的特征列
        fp_features = fp[feature_cols].copy()
        # 转换可能存在的非数值特征
        fp_features = convert_categorical_features(fp_features, feature_cols)

        fp_shap_values = explainer.shap_values(fp_features)

        # 绘制假阳性样本的SHAP摘要图
        # plt.figure(figsize=(12, 8))
        # shap.summary_plot(fp_shap_values, fp_features, feature_names=feature_cols, plot_size=(12, 8))
        # plt.title('假阳性(FP)样本的SHAP值摘要')
        # plt.tight_layout()
        # plt.savefig('error_analysis_plots/shap_fp_summary.png', dpi=300)
        # plt.close()

    # 计算假阴性样本的SHAP值
    if len(fn) > 0:
        # 只选择训练时使用的特征列
        fn_features = fn[feature_cols].copy()
        # 转换可能存在的非数值特征
        fn_features = convert_categorical_features(fn_features, feature_cols)

        fn_shap_values = explainer.shap_values(fn_features)

        # 绘制假阴性样本的SHAP摘要图
        # plt.figure(figsize=(12, 8))
        # shap.summary_plot(fn_shap_values, fn_features, feature_names=feature_cols, plot_size=(12, 8))
        # plt.title('假阴性(FN)样本的SHAP值摘要')
        # plt.tight_layout()
        # plt.savefig('error_analysis_plots/shap_fn_summary.png', dpi=300)
        # plt.close()

    print("\nSHAP值分析完成，已生成假阳性和假阴性样本的特征重要性图")

# ----------------------
# 7. 错误样本的共同特征分析
# ----------------------
def analyze_common_patterns(fp, fn, train_feature_names, top_n=5):
    """分析错误样本中最常见的特征模式"""
    # 使用训练时的特征列表
    feature_cols = train_feature_names

    # 分析假阳性样本的共同特征
    fp_patterns = {}
    if len(fp) > 0:
        # 对分类特征统计最常见值
        fp_features = fp[feature_cols].copy()
        cat_cols = fp_features.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            top_values = fp_features[col].value_counts().head(top_n)
            fp_patterns[col] = dict(top_values)

        # 对数值特征统计分布区间
        num_cols = fp_features.select_dtypes(include=['int64', 'float64']).columns
        for col in num_cols:
            # 计算分位数
            q25, q50, q75 = fp_features[col].quantile([0.25, 0.5, 0.75])
            fp_patterns[col] = f"中位数: {q50:.2f}, 四分位: [{q25:.2f}, {q75:.2f}]"

    # 分析假阴性样本的共同特征
    fn_patterns = {}
    if len(fn) > 0:
        # 对分类特征统计最常见值
        fn_features = fn[feature_cols].copy()
        cat_cols = fn_features.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            top_values = fn_features[col].value_counts().head(top_n)
            fn_patterns[col] = dict(top_values)

        # 对数值特征统计分布区间
        num_cols = fn_features.select_dtypes(include=['int64', 'float64']).columns
        for col in num_cols:
            # 计算分位数
            q25, q50, q75 = fn_features[col].quantile([0.25, 0.5, 0.75])
            fn_patterns[col] = f"中位数: {q50:.2f}, 四分位: [{q25:.2f}, {q75:.2f}]"

    # 保存分析结果
    import os
    if not os.path.exists('error_analysis_plots'):
        os.makedirs('error_analysis_plots')

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
    # 1. 加载数据和模型，获取训练时的特征列表
    model, combined_data, train_feature_names = load_data_and_model(
        model_path, test_data_path, prediction_path
    )

    # 转换分类特征为数值类型
    combined_data = convert_categorical_features(combined_data, train_feature_names)

    # 2. 分类错误样本并计算指标
    combined_data, tp, tn, fp, fn = categorize_errors(combined_data)

    # 3. 分析特征分布差异
    feature_stats = analyze_feature_distributions(
        combined_data, tp, tn, fp, fn, train_feature_names
    )

    # 4. 阈值分析
    threshold_df = analyze_threshold(combined_data)

    # 5. SHAP值分析
    shap_analysis(model, combined_data, fp, fn, train_feature_names)

    # 6. 常见模式分析
    fp_patterns, fn_patterns = analyze_common_patterns(fp, fn, train_feature_names)

    print("\n错误样本分析完成！所有结果已保存到 error_analysis_plots 目录")
    print("关键发现建议:")
    if len(fp) > 0:
        print(f"- 假阳性样本主要集中在 {list(fp_patterns.keys())[:3]} 等特征的特定区间")
    if len(fn) > 0:
        print(f"- 假阴性样本主要集中在 {list(fn_patterns.keys())[:3]} 等特征的特定区间")
    if not threshold_df.empty:
        print(f"- 建议尝试阈值 {threshold_df.loc[threshold_df['总错误数'].idxmin()]['阈值']} 以减少总错误数")

if __name__ == "__main__":
    # 配置文件路径
    MODEL_PATH = "credit_approval_model.txt"          # 模型文件路径
    TEST_DATA_PATH = "simulated_credit_data.csv"      # 测试数据路径（包含特征和真实标签）
    PREDICTION_PATH = "test_predictions.csv"          # 预测结果路径

    main(MODEL_PATH, TEST_DATA_PATH, PREDICTION_PATH)
