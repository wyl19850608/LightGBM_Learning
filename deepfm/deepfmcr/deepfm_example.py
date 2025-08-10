import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

# 1. 数据准备
# 这里使用示例数据，实际应用中请替换为你的数据集
def create_sample_data():
    """创建示例数据集"""
    np.random.seed(42)
    n_samples = 10000

    # 稀疏特征
    user_id = np.random.randint(1, 1000, n_samples)
    item_id = np.random.randint(1, 500, n_samples)
    category_id = np.random.randint(1, 20, n_samples)
    city = np.random.randint(1, 100, n_samples)

    #  dense特征
    age = np.random.randint(18, 60, n_samples)
    price = np.random.uniform(10, 1000, n_samples)
    click_count = np.random.randint(0, 100, n_samples)

    # 目标变量 (是否点击)
    target = np.random.randint(0, 2, n_samples)

    # 创建DataFrame
    data = pd.DataFrame({
        'user_id': user_id,
        'item_id': item_id,
        'category_id': category_id,
        'city': city,
        'age': age,
        'price': price,
        'click_count': click_count,
        'click': target
    })

    return data

# 获取示例数据
data = create_sample_data()
print("数据集基本信息：")
print(f"样本数量: {data.shape[0]}")
print(f"特征数量: {data.shape[1]-1}")  # 减去目标变量

# 2. 特征工程
# 划分稀疏特征和稠密特征
sparse_features = ['user_id', 'item_id', 'category_id', 'city']
dense_features = ['age', 'price', 'click_count']

# 缺失值填充
data[sparse_features] = data[sparse_features].fillna('-1',)
data[dense_features] = data[dense_features].fillna(0,)

# 1.Label Encoding for sparse features,and do simple Transformation for dense features
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])

# 2.对稠密特征做归一化
mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])

# 3.设置特征列
fixlen_feature_columns = [
                             SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4)
                             for feat in sparse_features
                         ] + [
                             DenseFeat(feat, 1,)
                             for feat in dense_features
                         ]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

# 4. 划分训练集和测试集
train, test = train_test_split(data, test_size=0.2, random_state=42)
train_model_input = {name: train[name].values for name in feature_names}
test_model_input = {name: test[name].values for name in feature_names}

# 5. 定义模型并训练
model = DeepFM(
    linear_feature_columns=linear_feature_columns,
    dnn_feature_columns=dnn_feature_columns,
    task='binary',  # 二分类任务
    dnn_hidden_units=(128, 128),  # DNN隐藏层
    l2_reg_linear=0.00001,
    l2_reg_embedding=0.00001,
    dnn_dropout=0.5,  # dropout防止过拟合
    seed=42
)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['AUC', 'accuracy']
)

print("\n开始训练模型...")
history = model.fit(
    train_model_input,
    train['click'].values,
    batch_size=256,
    epochs=10,
    verbose=1,
    validation_split=0.2  # 从训练集中划分20%作为验证集
)

# 6. 在测试集上评估
print("\n在测试集上评估模型...")
eval_result = model.evaluate(test_model_input, test['click'].values, verbose=0)
print(f"测试集结果 - 损失: {eval_result[0]:.4f}, AUC: {eval_result[1]:.4f}, 准确率: {eval_result[2]:.4f}")

# 7. 预测
print("\n进行预测...")
pred_ans = model.predict(test_model_input, batch_size=256)

# 8. 输出部分预测结果
print("\n部分预测结果:")
for i in range(10):
    print(f"实际值: {test['click'].values[i]}, 预测概率: {pred_ans[i][0]:.4f}")

# 9. 保存模型
model.save('deepfm_model.h5')
print("\n模型已保存为 deepfm_model.h5")
