import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.stat import Correlation, ChiSquareTest
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
import math

# 设置PySpark环境变量
os.environ["PYSPARK_PYTHON"] = "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10"

# 创建SparkSession
spark = SparkSession.builder \
    .appName("FeatureImportanceAnalysis") \
    .getOrCreate()

# 设置日志级别以减少警告信息
spark.sparkContext.setLogLevel("ERROR")

print("=== 特征重要性分析 ===")

# 假设已经有一个DataFrame，包含特征列和label列
# 如果需要读取数据，请取消下面的注释并替换为实际路径
# df = spark.read.csv("your_data.csv", header=True, inferSchema=True)

# 示例：创建一个简单的DataFrame用于演示
data = [
    (1, 10.5, "A", True, 1),
    (2, 20.3, "B", False, 0),
    (3, 15.7, "A", True, 1),
    (4, 5.2, "C", False, 0),
    (5, 25.1, "B", True, 1)
]

df = spark.createDataFrame(
    data,
    ["id", "numeric_feature", "string_feature", "boolean_feature", "label"]
)

# 定义特征重要性结果字典
feature_importance = {}

# 1. 连续特征的重要性（基于皮尔逊相关系数）
numeric_cols = [c for c, t in df.dtypes if t in ['int', 'double'] and c not in ['id', 'label']]

for col_name in numeric_cols:
    corr_value = abs(df.stat.corr(col_name, "label"))  # 使用绝对值
    feature_importance[col_name] = corr_value
    print(f"{col_name} 的重要性（相关系数绝对值）: {corr_value:.4f}")

# 2. 分类特征的重要性（基于卡方检验）
string_cols = [c for c, t in df.dtypes if t == 'string']
bool_cols = [c for c, t in df.dtypes if t == 'boolean']
cat_cols = string_cols + bool_cols

for col_name in cat_cols:
    # 对分类特征进行编码
    if col_name in string_cols:
        encoded_col = f"{col_name}_indexed"
        indexer = StringIndexer(inputCol=col_name, outputCol=encoded_col, handleInvalid='skip')
        df = indexer.fit(df).transform(df)
    else:  # 布尔列
        encoded_col = f"{col_name}_int"
        df = df.withColumn(encoded_col, F.col(col_name).cast("integer"))

    # 准备卡方检验数据
    chi_df = df.select(encoded_col, "label").dropna()

    # 转换为向量格式
    assembler = VectorAssembler(
        inputCols=[encoded_col],
        outputCol="features",
        handleInvalid="skip"
    )
    vector_df = assembler.transform(chi_df)

    # 执行卡方检验
    chi_result = ChiSquareTest.test(vector_df, "features", "label").head()
    p_value = chi_result.pValues[0]

    # 使用负对数p值作为重要性（p值越小，重要性越高）
    importance = -math.log(p_value) if p_value > 0 else float('inf')
    feature_importance[col_name] = importance
    print(f"{col_name} 的重要性（负对数p值）: {importance:.4f}")

# 3. 整合所有特征的重要性并排序
print("\n=== 特征重要性排序 ===")
sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

for feature, importance in sorted_importance:
    print(f"{feature}: {importance:.4f}")

# 4. 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.barh([x[0] for x in sorted_importance], [x[1] for x in sorted_importance])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance for Label')
plt.tight_layout()
plt.savefig("feature_importance.png")
print("特征重要性图已保存为: feature_importance.png")

# 5. 保存结果到CSV
importance_df = pd.DataFrame(sorted_importance, columns=['Feature', 'Importance'])
importance_df.to_csv('feature_importance.csv', index=False)
print("特征重要性数据已保存为: feature_importance.csv")

# 停止SparkSession
spark.stop()
print("\n=== 分析完成 ===")