import os
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan, when, count, rand, corr, skewness, kurtosis, lit
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.stat import Correlation, ChiSquareTest
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType, BooleanType

# 设置PySpark环境变量
os.environ["PYSPARK_PYTHON"] = "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10"

# 创建SparkSession
spark = SparkSession.builder \
    .appName("DataAnalysis") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .getOrCreate()

# 设置日志级别以减少警告信息
spark.sparkContext.setLogLevel("ERROR")

print("=== 数据生成与准备 ===")
# 定义可能的字符串类别
categories = ["Aa", "Bb", "Cc", "Dd", "Ee"]

# 完全使用Python原生随机数生成，避免NumPy
import random

# 生成示例数据，使用Python原生类型
# data = [
#     (
#         i,                          # id: 整数
#         random.normalvariate(0, 1),  # feature1: 浮点数（Python float）
#         random.normalvariate(0, 1),  # feature2: 浮点数（Python float）
#         random.randint(1, 5),        # category: 整数
#         random.choice(categories),  # string_col: 字符串
#         random.choice([True, False]) # boolean_col: 布尔值
#     )
#     for i in range(100)
# ]

null_probabilities = {
    "id": 0.0,          # ID列不设置空值
    "feature1": 0.30,   # 5%概率为空
    "feature2": 0.03,   # 3%概率为空
    "category": 0.02,   # 2%概率为空
    "string_col": 0.10, # 10%概率为空
    "boolean_col": 0.08 # 8%概率为空
}

# 生成示例数据，随机添加空值
data = []
for i in range(1000):
    # 生成各列值，根据概率决定是否设为空
    id_val = i
    feature1_val = random.normalvariate(0, 1) if random.random() > null_probabilities["feature1"] else None
    feature2_val = random.normalvariate(0, 1) if random.random() > null_probabilities["feature2"] else None
    category_val = random.randint(1, 5) if random.random() > null_probabilities["category"] else None
    string_val = random.choice(categories) if random.random() > null_probabilities["string_col"] else None
    boolean_val = random.choice([True, False]) if random.random() > null_probabilities["boolean_col"] else None

    data.append((id_val, feature1_val, feature2_val, category_val, string_val, boolean_val))




# 定义Schema，明确各列类型
schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("feature1", DoubleType(), True),
    StructField("feature2", DoubleType(), True),
    StructField("category", IntegerType(), True),
    StructField("string_col", StringType(), True),
    StructField("boolean_col", BooleanType(), True)
])

# 创建DataFrame
df = spark.createDataFrame(data, schema)

# 添加随机label列(0/1)，使用Spark原生函数
df = df.withColumn("label", (rand() > 0.5).cast("integer"))

print("\n=== 数据基本信息 ===")
print(f"数据行数: {df.count()}")
print(f"数据列数: {len(df.columns)}")
df.printSchema()
df.show(5, truncate=False)

print("\n=== 缺失值分析 ===")
# 区分处理数值型和非数值型列
missing_counts = df.select([
    (count(when(isnan(c) | col(c).isNull(), c)) if t in ['double', 'float', 'int']
     else count(when(col(c).isNull(), c))).alias(c)
    for c, t in df.dtypes
])

missing_counts.show()

missing_pct = missing_counts.select([(col(c) / df.count() * 100).alias(f"{c}_missing_pct") for c in missing_counts.columns])
missing_pct.show()



print("\n=== 基本统计量分析 ===")
# 计算数值列的统计量
numeric_cols = [c for c, t in df.dtypes if t in ['int', 'double']]
summary = df.select(numeric_cols).summary("count", "mean", "stddev", "min", "max")
summary.show()

print("\n=== 4. 相关性矩阵可视化 ===")
# 筛选数值型列（排除id）
numeric_cols = [c for c, t in df.dtypes if t in ['int', 'double'] and c not in ['id']]

# 转换为MLlib向量格式
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features",handleInvalid="skip")
vector_df = assembler.transform(df).select("features")

# 计算相关矩阵（皮尔逊）
corr_matrix = Correlation.corr(vector_df, "features").collect()[0][0]

# 转换为列表结构，避免使用NumPy
corr_list = [[float(corr_matrix[i, j]) for j in range(corr_matrix.numCols)] for i in range(corr_matrix.numRows)]

# 使用Pandas和Matplotlib可视化
plt.figure(figsize=(10, 8))
corr_df = pd.DataFrame(corr_list, columns=numeric_cols, index=numeric_cols)
plt.imshow(corr_df.values, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=45)
plt.yticks(range(len(corr_df.columns)), corr_df.columns)
plt.title("Correlation Matrix (Pearson)")
plt.tight_layout()
plt.savefig("correlation_matrix_pearson.png")
print("皮尔逊相关性矩阵已保存为: correlation_matrix_pearson.png")

print("\n=== 5. 字符串型特征与标签的相关性 ===")
# 对字符串列进行索引编码
string_cols = [c for c, t in df.dtypes if t == 'string']
for col_name in string_cols:
    indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_indexed",handleInvalid='skip')
    df = indexer.fit(df).transform(df)

    # 计算与label的相关性（使用编码后的值）
    corr_value = df.stat.corr(f"{col_name}_indexed", "label")
    print(f"{col_name} 与 label 的相关性: {corr_value:.4f}")

    # 将索引列转换为向量列
    vectorize = udf(lambda x: Vectors.dense([x]), VectorUDT())
    df = df.withColumn(f"{col_name}_vector", vectorize(f"{col_name}_indexed"))

    # 卡方检验（使用向量列）
    chi_result = ChiSquareTest.test(df, f"{col_name}_vector", "label").head()

    # 根据Spark版本获取p值
    if hasattr(chi_result, "pValues"):
        # 新版本Spark返回pValues向量
        p_value = chi_result.pValues[0]
    else:
        # 旧版本Spark返回单个pValue
        p_value = chi_result.pValue

    print(f"{col_name} 与 label 的卡方检验 p值: {p_value:.4f}")

print("\n=== 6. 布尔型特征与标签的相关性 ===")
# 布尔型特征处理
bool_cols = [c for c, t in df.dtypes if t == 'boolean']
for col_name in bool_cols:
    # 将布尔值转换为整数(0/1)
    df = df.withColumn(f"{col_name}_int", col(col_name).cast("integer"))

    # 过滤掉包含缺失值的行
    filtered_df = df.filter(col(f"{col_name}_int").isNotNull() & col("label").isNotNull())


    # 计算与label的相关性
    corr_value = filtered_df.stat.corr(f"{col_name}_int", "label")
    print(f"{col_name} 与 label 的相关性: {corr_value:.4f}")

    # 将整数列转换为向量列
    vectorize = udf(lambda x: Vectors.dense([x]), VectorUDT())
    df = df.withColumn(f"{col_name}_vector", vectorize(f"{col_name}_int"))

    # 卡方检验（使用向量列）
    chi_result = ChiSquareTest.test(df, f"{col_name}_vector", "label").head()

    # 根据Spark版本获取p值
    if hasattr(chi_result, "pValues"):
        p_value = chi_result.pValues[0]
    else:
        p_value = chi_result.pValue

    print(f"{col_name} 与 label 的卡方检验 p值: {p_value:.4f}")

print("\n=== 7. 类别型特征的分布分析 ===")
# 类别型特征分布
cat_cols = string_cols + bool_cols + ['category']
for col_name in cat_cols:
    print(f"\n{col_name} 的分布:")
    df.groupBy(col_name).count().show()

print("\n=== 8. 特征与标签的交叉表分析 ===")
# 创建交叉表
for col_name in string_cols + bool_cols:
    print(f"\n{col_name} 与 label 的交叉表:")
    df.crosstab(col_name, "label").show()

# 停止SparkSession
spark.stop()
print("\n=== 分析完成 ===")