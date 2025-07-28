import numpy as np
import pandas as pd

# 假设 loaded_encoders 是从训练中加载的编码器字典，每个编码器包含 classes_ 属性
# 示例：loaded_encoders = {'gender': LabelEncoder(), 'education': LabelEncoder(), ...}
# 这里为了演示，模拟一个训练好的编码器
class MockEncoder:
    def __init__(self, classes):
        self.classes_ = np.array(classes)

# 模拟训练时的编码器（假设训练数据中见过的类别）
loaded_encoders = {
    'gender': MockEncoder(['男', '女']),
    'education': MockEncoder(['本科', '硕士', '博士', '高中']),
    'city': MockEncoder(['北京', '上海', '广州', '深圳'])
}

# 新数据（修复了长度不一致的问题）
new_data = {
    'user_id': [1011, 1012, 1013, 1014, 1015],
    'gender': ['男', '女', '女', '男', '2222'],  # 包含新类别'2222'
    'education': ['本科', '硕士', '博士', '高中', '大专'],  # 包含新类别'大专'
    'city': ['北京', '上海', '广州', '深圳', '杭州'],  # 包含新类别'杭州'
    'score': [0.72, 0.85, 0.68, 0.91, 0.79]
}
df_new = pd.DataFrame(new_data)
cat_cols = ['gender', 'education', 'city']  # 分类特征列

print("111"*20)
print(df_new.dtypes)
print("0000"*20)

for col in cat_cols:
    if col in loaded_encoders:
        # 获取训练时见过的所有类别及对应的编码
        encoder = loaded_encoders[col]
        seen_categories = set(encoder.classes_)
        # 创建类别到编码的映射字典
        cat_to_code = {cat: idx for idx, cat in enumerate(encoder.classes_)}

        print(f"\n{col} 训练时见过的类别及编码: {cat_to_code}")

        # 将新数据转换为字符串，并处理未见过的类别
        df_new[col] = df_new[col].astype(str)
        # 统计未见过的类别
        unseen = set(df_new[col]) - seen_categories
        if unseen:
            print(f"  发现未见过的类别: {unseen}，将替换为'unknown'")
            df_new[col] = df_new[col].apply(lambda x: x if x in seen_categories else 'unknown')

        # 确保'unknown'在编码映射中（如果没有则添加）
        if 'unknown' not in cat_to_code:
            # 将'unknown'编码为最大值+1
            max_code = len(cat_to_code)
            cat_to_code['unknown'] = max_code
            print(f"  添加'unknown'的编码: {max_code}")

        # 应用编码（使用映射字典手动转换，而非transform）
        df_new[col] = df_new[col].map(cat_to_code)
        print(df_new[col].dtype)

print("\n处理后的新数据:")
print(df_new)

print("22222"*20)
print(df_new.dtypes)
print("33333"*20)
