import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score

# 生成示例数据集
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=0,
    random_state=42
)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建 LightGBM 数据集
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 设置参数
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'verbosity': -1,
    'seed': 42
}

# 训练模型
num_round = 100
model = lgb.train(
    params,
    train_data,
    num_round,
    valid_sets=[test_data],
    early_stopping_rounds=10,
    verbose_eval=False
)

# 预测
y_pred_proba = model.predict(X_test)
y_pred = [1 if x >= 0.5 else 0 for x in y_pred_proba]

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 输出结果
print(f"准确率: {accuracy:.4f}")
print(f"召回率: {recall:.4f}")
print(f"F1 分数: {f1:.4f}")