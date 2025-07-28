import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False

# ----------------------
# 1. 数据加载与预处理
# ----------------------
def load_and_preprocess_data(file_path):
    """加载数据并进行预处理"""
    df = pd.read_csv(file_path)

    # 处理日期字段
    date_cols = ['cust_recv_time', 'touch_time']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # 处理缺失值
    # 数值型字段用0填充
    num_cols = ['crdt_lim_yx', 'pril_bal', 'total_loan_cnt', 'total_loan_amt',
                'apply_cnt', 'wdraw_cnt', 'push_cnt', 'sms_charge_cnt']
    for col in num_cols:
        if col in df.columns:
            # 先记录缺失标识，再填充
            df[f'{col}_is_null'] = df[col].isna().astype(int)
            df[col] = df[col].fillna(0)

    # 类别型字段用特殊值填充
    cat_cols = ['gender_cd', 'residence_pr_cd', 'occ_cd', 'if_bj_10_yn',
                'if_bj_30_yn', 'is_coupon_issue']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    # 处理时间差特征
    if 'touch_time' in df.columns and 'cust_recv_time' in df.columns:
        df['recv_to_touch_days'] = (df['touch_time'] - df['cust_recv_time']).dt.days
        df['recv_to_touch_days'] = df['recv_to_touch_days'].fillna(-1)  # 缺失用-1表示

    return df

# ----------------------
# 2. 特征工程
# ----------------------
def create_features(df):
    """创建特征工程"""
    features = df.copy()

    # 额度相关特征
    if 'crdt_lim_yx' in features.columns:
        # 额度等级 (分箱)
        features['lim_level'] = pd.cut(
            features['crdt_lim_yx'],
            bins=[0, 10000, 20000, 30000, float('inf')],
            labels=[1, 2, 3, 4]
        ).astype(float)
        # 额度是否为0
        features['is_zero_limit'] = (features['crdt_lim_yx'] == 0).astype(int)

    # 行为比率特征
    if 'wdraw_cnt' in features.columns and 'apply_cnt' in features.columns:
        # 提现成功率 (避免除零)
        features['wdraw_success_ratio'] = np.where(
            features['apply_cnt'] > 0,
            features['wdraw_cnt'] / features['apply_cnt'],
            0
        )
        # 未提现申请占比
        features['unwdraw_ratio'] = np.where(
            features['apply_cnt'] > 0,
            (features['apply_cnt'] - features['wdraw_cnt']) / features['apply_cnt'],
            0
        )

    # 贷款频率特征
    if 'total_loan_cnt' in features.columns:
        # 贷款活跃度 (是否有过贷款)
        features['has_loan'] = (features['total_loan_cnt'] > 0).astype(int)
        # 平均每次贷款金额
        features['avg_loan_amt'] = np.where(
            features['total_loan_cnt'] > 0,
            features['total_loan_amt'] / features['total_loan_cnt'],
            0
        )
        # 贷款金额与额度比率
        if 'crdt_lim_yx' in features.columns:
            features['loan_to_limit_ratio'] = np.where(
                features['crdt_lim_yx'] > 0,
                features['total_loan_amt'] / features['crdt_lim_yx'],
                0
            )

    # 营销响应特征
    if 'push_cnt' in features.columns and 'sms_charge_cnt' in features.columns:
        # 总触达次数
        features['total_touch_cnt'] = features['push_cnt'] + features['sms_charge_cnt']
        # 触达响应率
        features['touch_response_ratio'] = np.where(
            features['total_touch_cnt'] > 0,
            features['wdraw_cnt'] / features['total_touch_cnt'],
            0
        )
        # 短信触达占比
        features['sms_touch_ratio'] = np.where(
            features['total_touch_cnt'] > 0,
            features['sms_charge_cnt'] / features['total_touch_cnt'],
            0
        )

    # 客群特征 - 职业与额度交叉
    if 'occ_cd' in features.columns and 'crdt_lim_yx' in features.columns:
        features['occ_lim_interaction'] = features['occ_cd'].astype(str) + '_' + (features['crdt_lim_yx'] // 10000).astype(str)

    return features

# ----------------------
# 3. 数据划分与特征处理
# ----------------------
def prepare_training_data(features):
    """准备训练数据"""
    # 定义类别特征和数值特征
    cat_features = ['gender_cd', 'residence_pr_cd', 'occ_cd', 'if_bj_10_yn',
                    'if_bj_30_yn', 'is_coupon_issue', 'lim_level', 'occ_lim_interaction']
    cat_features = [col for col in cat_features if col in features.columns]

    # 类别特征编码
    for col in cat_features:
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col].astype(str))

    # 筛选特征列 (排除ID、时间和目标变量)
    exclude_cols = ['target', 'cust_recv_time', 'touch_time', 'cust_id']  # 增加可能的ID字段排除
    feature_cols = [col for col in features.columns if col not in exclude_cols]

    # 按时间排序，确保时间序列划分合理性
    if 'cust_recv_time' in features.columns:
        features = features.sort_values('cust_recv_time')

    # 划分训练集和测试集 (时间序列划分)
    train_size = int(0.7 * len(features))
    train = features.iloc[:train_size]
    test = features.iloc[train_size:]

    # 提取特征和目标变量
    X_train, y_train = train[feature_cols], train['target']
    X_test, y_test = test[feature_cols], test['target']

    print(f"训练集样本数: {len(X_train)}, 正例比例: {y_train.mean():.2%}")
    print(f"测试集样本数: {len(X_test)}, 正例比例: {y_test.mean():.2%}")
    print(f"特征数量: {len(feature_cols)}, 类别特征数量: {len(cat_features)}")

    return X_train, X_test, y_train, y_test, feature_cols, cat_features, test[['cust_recv_time', 'touch_time']]  # 返回时间列用于后续分析

# ----------------------
# 4. 模型训练与评估
# ----------------------
def train_evaluate_model(X_train, X_test, y_train, y_test, feature_cols, cat_features, time_cols):
    """训练并评估LightGBM模型"""
    # 构建LightGBM数据集
    lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    lgb_test = lgb.Dataset(X_test, label=y_test, reference=lgb_train, categorical_feature=cat_features)

    # 计算正负样本比例，用于处理不平衡问题
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    pos_weight = neg_count / pos_count if pos_count > 0 else 1

    # 设置模型参数
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 5,
        'min_data_in_leaf': 50,
        'reg_alpha': 0.1,
        'reg_lambda': 0.2,
        'scale_pos_weight': pos_weight,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }

    # 创建回调函数
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),  # 早停回调
        lgb.log_evaluation(period=100)  # 每100轮打印一次日志
    ]

    # 训练模型
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_test],
        callbacks=callbacks
    )

    # 预测
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba >= 0.5).astype(int)  # 默认阈值0.5

    # 评估指标
    metrics = {
        'AUC': roc_auc_score(y_test, y_pred_proba),
        '精确率': precision_score(y_test, y_pred),
        '召回率': recall_score(y_test, y_pred),
        'F1分数': f1_score(y_test, y_pred)
    }

    print("\n模型评估指标:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # 混淆矩阵


    # 返回包含时间信息的预测结果
    test_results = pd.concat([X_test.reset_index(drop=True),
                              pd.DataFrame({
                                  '实际标签': y_test.reset_index(drop=True),
                                  '预测概率': y_pred_proba,
                                  '预测标签': y_pred
                              })], axis=1)

    test_results.to_csv('test_predictions_with_features.csv', index=False)
    print("测试集预测结果（含原始特征）已保存为: test_predictions_with_features.csv")

    return model, metrics, test_results

# ----------------------
# 5. 主函数
# ----------------------
def main(data_path):
    # 1. 加载并预处理数据
    df = load_and_preprocess_data(data_path)

    # 2. 特征工程
    features = create_features(df)

    # 3. 准备训练数据
    X_train, X_test, y_train, y_test, feature_cols, cat_features, time_cols = prepare_training_data(features)

    # 4. 训练并评估模型
    model, metrics, test_results = train_evaluate_model(
        X_train, X_test, y_train, y_test, feature_cols, cat_features, time_cols
    )

    # 5. 保存模型和结果
    model.save_model('credit_approval_model.txt')
    print("\n模型已保存为: credit_approval_model.txt")

    # 保存测试集预测结果（包含原始特征、时间、标签和预测值）
    # test_results.to_csv('test_predictions_with_features.csv', index=False)
    # print("测试集预测结果（含原始特征）已保存为: test_predictions_with_features.csv")

    return model, metrics

if __name__ == "__main__":
    # 数据路径（可替换为实际数据路径）
    data_path = "simulated_credit_data.csv"
    model, metrics = main(data_path)