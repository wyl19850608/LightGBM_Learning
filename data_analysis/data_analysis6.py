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

null_probabilities = {
    "id": 0.0,          # ID列不设置空值
    "feature1": 0.30,   # 30%概率为空
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
missing_counts = []
for c, t in df.dtypes:
    if t in ['double', 'float', 'int']:
        count_expr = count(when(isnan(c) | col(c).isNull(), c)).alias(c)
    else:
        count_expr = count(when(col(c).isNull(), c)).alias(c)
    count_result = df.select(count_expr).collect()[0][0]
    missing_counts.append(count_result)

missing_counts_df = spark.createDataFrame([missing_counts], df.columns)
missing_counts_df.show()

missing_pct = []
for c in df.columns:
    pct_expr = (col(c) / df.count() * 100).alias(f"{c}_missing_pct")
    pct_result = missing_counts_df.select(pct_expr).collect()[0][0]
    missing_pct.append(pct_result)

missing_pct_df = spark.createDataFrame([missing_pct], [f"{c}_missing_pct" for c in df.columns])
missing_pct_df.show()

print("\n=== 基本统计量分析 ===")
numeric_cols = [c for c, t in df.dtypes if t in ['int', 'double']]
for col_name in numeric_cols:
    summary = df.select(col_name).summary("count", "mean", "stddev", "min", "max")
    print(f"列 {col_name} 的统计量:")
    summary.show()

print("\n=== 4. 相关性矩阵可视化 ===")
numeric_cols = [c for c, t in df.dtypes if t in ['int', 'double'] and c not in ['id']]
corr_matrix = []
for i in range(len(numeric_cols)):
    row = []
    for j in range(len(numeric_cols)):
        if i == j:
            row.append(1.0)
        else:
            corr_value = df.stat.corr(numeric_cols[i], numeric_cols[j])
            row.append(corr_value)
    corr_matrix.append(row)

# 使用Pandas和Matplotlib可视化
plt.figure(figsize=(10, 8))
corr_df = pd.DataFrame(corr_matrix, columns=numeric_cols, index=numeric_cols)
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
    indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_indexed", handleInvalid='skip')
    df = indexer.fit(df).transform(df)

    # 计算与label的相关性（使用编码后的值）
    corr_value = df.stat.corr(f"{col_name}_indexed", "label")
    print(f"{col_name} 与 label 的相关性: {corr_value:.4f}")

    # 将索引列转换为向量列
    vectorize = udf(lambda x: Vectors.dense([x]), VectorUDT())
    df = df.withColumn(f"{col_name}_vector", vectorize(f"{col_name}_indexed"))

    # 创建仅包含所需列的DataFrame并移除缺失值
    chi_df = df.select(f"{col_name}_vector", "label").dropna()

    # 卡方检验（使用过滤后的DataFrame）
    chi_result = ChiSquareTest.test(chi_df, f"{col_name}_vector", "label").head()

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

    # 创建仅用于卡方检验的DataFrame，确保无缺失值
    chi_df = df.select(f"{col_name}_int", "label").dropna()

    # 计算与label的相关性
    corr_value = chi_df.stat.corr(f"{col_name}_int", "label")
    print(f"{col_name} 与 label 的相关性: {corr_value:.4f}")

    # 使用VectorAssembler创建向量列，自动处理缺失值
    assembler = VectorAssembler(
        inputCols=[f"{col_name}_int"],
        outputCol=f"{col_name}_vector",
        handleInvalid="skip"
    )
    vector_df = assembler.transform(chi_df)

    # 卡方检验
    chi_result = ChiSquareTest.test(vector_df, f"{col_name}_vector", "label").head()

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