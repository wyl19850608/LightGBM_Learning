import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

# ----------------------
# 1. 数据准备（示例数据）
# ----------------------
# 生成模拟数据（包含多种类型特征和缺失值）
np.random.seed(42)
n_samples = 10000

# 连续型特征（含缺失值）
data = {
    'age': np.random.normal(30, 10, n_samples).round().clip(0, 100),  # 年龄
    'income': np.random.lognormal(10, 0.5, n_samples).round(),  # 收入
    'score': np.random.normal(50, 15, n_samples).round().clip(0, 100)  # 评分
}

# 类别型特征（含缺失值）
data['education'] = np.random.choice(['high', 'college', 'graduate', 'phd'], n_samples)  # 教育程度
data['city'] = np.random.choice(['beijing', 'shanghai', 'guangzhou', 'shenzhen'], n_samples)  # 城市
data['is_married'] = np.random.choice([0, 1], n_samples)  # 婚姻状态（布尔型）

# 手动添加缺失值（模拟真实场景）
for col in ['age', 'income']:
    data[col][np.random.choice(n_samples, int(n_samples*0.2))] = np.nan  # 20%缺失
data['education'][np.random.choice(n_samples, int(n_samples*0.3))] = np.nan  # 30%缺失

# 标签（二分类：是否购买）
data['label'] = np.where(
    (data['income'] > np.median(data['income'])) & (data['age'] > 30),
    1, 0
)

# 转换为DataFrame
df = pd.DataFrame(data)

# 划分特征和标签
X = df.drop('label', axis=1)
y = df['label']

# ----------------------
# 2. 数据预处理（核心逻辑）
# ----------------------
class LGBMDataProcessor:
    def __init__(self, missing_threshold=0.3):
        self.missing_threshold = missing_threshold  # 缺失值比例阈值（超过则删除特征）
        self.num_cols = None  # 连续特征列名
        self.cat_cols = None  # 类别特征列名
        self.scaler = StandardScaler()  # 标准化器（用于连续特征）
        self.kept_features = None  # 保留的特征列名
        self.imputers = {}  # 存储每个特征的填充值（众数/特殊值）

    def fit(self, X):
        # 步骤1：筛选高缺失值特征
        missing_ratio = X.isnull().mean()
        self.kept_features = [col for col in X.columns if missing_ratio[col] < self.missing_threshold]
        X_filtered = X[self.kept_features].copy()

        # 步骤2：区分连续和类别特征
        self.num_cols = X_filtered.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.cat_cols = [col for col in X_filtered.columns if col not in self.num_cols]

        # 步骤3：缺失值处理
        # 连续特征：用众数填充（对异常值更稳健，比均值好）
        for col in self.num_cols:
            self.imputers[col] = X_filtered[col].mode()[0]  # 众数
        # 类别特征：用特殊值"Unknown"填充
        for col in self.cat_cols:
            self.imputers[col] = 'Unknown'  # 特殊标记

        # 步骤4：连续特征标准化（可选，树模型不强制，但有助于稳定训练）
        X_num_imputed = X_filtered[self.num_cols].fillna(self.imputers)
        self.scaler.fit(X_num_imputed)

        return self

    def transform(self, X):
        # 筛选保留的特征
        X_filtered = X[self.kept_features].copy() if self.kept_features else X.copy()

        # 缺失值填充
        X_filtered[self.num_cols] = X_filtered[self.num_cols].fillna(self.imputers)
        X_filtered[self.cat_cols] = X_filtered[self.cat_cols].fillna(self.imputers)

        # 连续特征标准化
        X_filtered[self.num_cols] = self.scaler.transform(X_filtered[self.num_cols])

        # 类别特征转换为字符串（LightGBM要求类别特征为字符串类型）
        for col in self.cat_cols:
            X_filtered[col] = X_filtered[col].astype(str)

        return X_filtered

    def get_feature_types(self):
        # 返回LightGBM所需的类别特征索引（用于训练时指定）
        if not self.kept_features:
            return []
        return [i for i, col in enumerate(self.kept_features) if col in self.cat_cols]

# 实例化处理器并拟合
processor = LGBMDataProcessor(missing_threshold=0.3)
processor.fit(X)

# 转换数据（训练集）
X_processed = processor.transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# ----------------------
# 3. 特征筛选（基于重要性）
# ----------------------
# 先训练一个基础模型筛选特征
base_model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42
)
base_model.fit(
    X_train, y_train,
    categorical_feature=processor.get_feature_types(),
    verbose=False
)

# 基于特征重要性筛选（保留重要性>0的特征）
feature_importance = pd.Series(
    base_model.feature_importances_,
    index=X_train.columns
)
selected_features = feature_importance[feature_importance > 0].index.tolist()

# 过滤特征
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# 更新类别特征索引（筛选后）
cat_indices = [i for i, col in enumerate(selected_features) if col in processor.cat_cols]

# ----------------------
# 4. 模型训练与评估
# ----------------------
# 优化后的LightGBM参数
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 63,  # 控制复杂度（2^max_depth附近）
    'max_depth': 6,    # 限制树深，防止过拟合
    'min_data_in_leaf': 100,  # 叶子最小样本数
    'reg_alpha': 0.1,  # L1正则化
    'reg_lambda': 0.1, # L2正则化
    'feature_fraction': 0.8,  # 特征采样
    'bagging_fraction': 0.8,  # 样本采样
    'bagging_freq': 5,  # 每5轮采样一次
    'is_unbalanced': False,  # 若样本不平衡，设为True
    'verbose': -1,
    'random_state': 42
}

# 划分验证集（用于早停）
X_train, X_val, y_train, y_val = train_test_split(
    X_train_selected, y_train, test_size=0.1, random_state=42
)

# 训练模型（带早停）
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_indices)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, val_data],
    num_boost_round=1000,
    early_stopping_rounds=50,  # 早停（50轮无提升则停止）
    verbose_eval=100
)

# ----------------------
# 5. 模型评估
# ----------------------
# 测试集预测
y_pred_proba = model.predict(X_test_selected, num_iteration=model.best_iteration)
y_pred = np.round(y_pred_proba).astype(int)  # 二分类阈值0.5

# 评估指标
print(f"测试集准确率: {accuracy_score(y_test, y_pred):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 特征重要性可视化
lgb.plot_importance(model, importance_type='gain', title='特征重要性（增益）')
plt.show()