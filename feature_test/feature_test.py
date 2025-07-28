import pandas as pd
import numpy as np

# 创建示例数据
data = {
    "category_str": ["A", "B", "C", "A", "B", None, "C", "A"],
    "category_int": [10, 20, 30, 20, 10, 30, 20, 10],
    "mixed_type": ["X", "Y", 0, 1, None, "X", 0, "Y"]
}
df = pd.DataFrame(data)
categorical_features = ["category_str", "category_int", "mixed_type"]

# 保存转换前的数据和类型
before_data = df.copy()
before_types = {col: str(before_data[col].dtype) for col in categorical_features}

# 执行转换操作
for col in categorical_features:
    df[col] = df[col].astype('category')

# 转换后的类型
after_types = {col: str(df[col].dtype) for col in categorical_features}

# 打印转换前的信息
print("=== 转换前 ===")
print("数据类型:")
for col, dtype in before_types.items():
    print(f"- {col}: {dtype}")
print("\n数据内容:")
print(before_data)

# 打印转换后的信息
print("\n=== 转换后 ===")
print("数据类型:")
for col, dtype in after_types.items():
    print(f"- {col}: {dtype}")
print("\n数据内容(表面值):")
print(df)

# 打印分类编码信息（内部存储）
print("\n=== 分类编码详情（内部存储） ===")
for col in categorical_features:
    print(f"\n{col}的分类值: {df[col].cat.categories.tolist()}")
    print(f"{col}的整数编码: {df[col].cat.codes.tolist()}")
    print(f"缺失值位置: {df[col].isna().tolist()}")
