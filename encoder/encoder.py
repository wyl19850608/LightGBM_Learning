import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# ----------------------
# 1. 准备示例数据
# ----------------------
# 创建包含分类特征的数据集
data = {
    'user_id': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010],
    'gender': ['男', '女', '男', '男', '女', '女', '男', '女', '男', '女'],
    'education': ['本科', '硕士', '高中', '本科', '博士', '本科', '硕士', '高中', '本科', '硕士'],
    'city': ['北京', '上海', '广州', '北京', '上海', '深圳', '广州', '北京', '上海', '深圳'],
    'score': [0.78, 0.65, 0.92, 0.83, 0.71, 0.88, 0.75, 0.69, 0.81, 0.77]
}

df_train = pd.DataFrame(data)
print("原始训练数据:")
print(df_train)
print("\n各分类列的原始值:")
for col in ['gender', 'education', 'city']:
    print(f"{col}: {df_train[col].unique()}")

# ----------------------
# 2. 训练并保存编码器
# ----------------------
# 定义需要编码的分类列
cat_cols = ['gender', 'education', 'city']

# 初始化编码器字典
encoders = {}

# 对每个分类列进行编码
for col in cat_cols:
    le = LabelEncoder()
    # 训练编码器并转换数据（先转为字符串处理，避免类型问题）
    df_train[col] = le.fit_transform(df_train[col].astype(str))
    # 保存编码器
    encoders[col] = le
    # 打印编码映射关系
    print(f"\n{col} 编码映射:")
    for cls, code in zip(le.classes_, le.transform(le.classes_)):
        print(f"  {cls} → {code}")

print("\n编码后的训练数据:")
print(df_train)

# 保存编码器到文件
save_path = 'encoders.pkl'
joblib.dump(encoders, save_path)
print(f"\n编码器已保存到: {os.path.abspath(save_path)}")

# ----------------------
# 3. 模拟新场景：加载编码器并处理新数据
# ----------------------
print("\n===== 处理新数据 =====")

# 准备新数据（包含已知类别和新类别）
new_data = {
    'user_id': [1011, 1012, 1013, 1014, 1015],
    'gender': ['男', '女', '女', '男', '未知'],  # 包含新类别'未知'
    'education': ['本科', '硕士', '博士', '高中', '大专'],  # 包含新类别'大专'
    'city': ['北京', '上海', '广州', '深圳', '杭州'],  # 包含新类别'杭州'
    'score': [0.72, 0.85, 0.68, 0.91, 0.79]
}

df_new = pd.DataFrame(new_data)
print("原始新数据:")
print(df_new)

# 加载保存的编码器
loaded_encoders = joblib.load(save_path)
print(f"\n已从 {os.path.abspath(save_path)} 加载编码器")


new_data = {
    'user_id': [1011, 1012, 1013, 1014, 1015,1016],
    'gender': ['男', '女', '女', '男',  '2222','未知'],  # 包含新类别'未知'
    'education': ['本科', '硕士', '博士', '高中', '',''],  # 包含新类别'大专'
    'city': ['北京', '上海', '广州', '深圳', '杭州', "",],  # 包含新类别'杭州'
    'score': [0.72, 0.85, 0.68, 0.91, 0.79, 0.8]
}

df_new = pd.DataFrame(new_data)


print("111"*20)
print(df_new.dtypes)
print("0000"*20)

# 处理新数据
for col in cat_cols:
    if col in loaded_encoders:
        # 获取训练时见过的所有类别
        seen_categories = set(loaded_encoders[col].classes_)
        print(f"\n{col} 训练时见过的类别: {seen_categories}")

        # 将新数据转换为字符串，并处理未见过的类别
        df_new[col] = df_new[col].astype(str)
        # 统计未见过的类别
        unseen = set(df_new[col]) - seen_categories
        if unseen:
            print(f"  发现未见过的类别: {unseen}，将替换为'unknown'")
            df_new[col] = df_new[col].apply(lambda x: x if x in seen_categories else 'unknown')

        # 确保'unknown'在编码器中有对应编码（如果没有则添加）
        if 'unknown' not in seen_categories:
            # 获取当前最大编码值，将'unknown'编码为最大值+1
            max_code = len(loaded_encoders[col].classes_)
            print(f"  添加'unknown'的编码: {max_code}")
            # 重新训练编码器以包含'unknown'（仅在首次遇到时）
            new_classes = list(loaded_encoders[col].classes_) + ['unknown']
            loaded_encoders[col].classes_ = np.array(new_classes)

        # 应用编码
        df_new[col] = loaded_encoders[col].transform(df_new[col])

print("2"*20)
print(df_new.dtypes)
print("3"*20)
print("\n编码后的新数据:")
print(df_new)
