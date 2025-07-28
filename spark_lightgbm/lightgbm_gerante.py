# train_lightgbm.py
import lightgbm as lgb
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# ----------------------
# 1. 生成模拟训练数据
# ----------------------
# 生成二分类数据：10000样本，20特征（5分类特征，15数值特征）
X, y = make_classification(
    n_samples=10000, n_features=20, n_informative=10,
    random_state=42, n_classes=2
)

# 构造DataFrame，前5列设为分类特征（字符串类型），后15列为数值特征
df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(20)])
# 将前5列转为分类字符串（如 "cat_0", "cat_1"）
for i in range(5):
    df[f'feat_{i}'] = pd.cut(
        df[f'feat_{i}'], bins=5, labels=[f'cat_{j}' for j in range(5)]
    )
# 添加标签列
df['label'] = y

# 划分训练集
X_train = df.drop('label', axis=1)
y_train = df['label']
categorical_features = [f'feat_{i}' for i in range(5)]  # 前5列为分类特征

# ----------------------
# 2. 训练LightGBM模型
# ----------------------
train_data = lgb.Dataset(
    X_train,
    label=y_train,
    categorical_feature=categorical_features  # 指定分类特征
)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[train_data],
    valid_names=['train']
)

# ----------------------
# 3. 保存模型（供Spark加载）
# ----------------------
model.save_model('lightgbm_model.txt')  # 保存为原生txt格式
print("模型保存路径：lightgbm_model.txt")
print("特征名称：", model.feature_name())  # 打印特征名（预测时需匹配）