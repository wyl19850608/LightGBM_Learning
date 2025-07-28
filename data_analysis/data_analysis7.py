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
    "string_col1": 0.10, # 10%概率为空
    "boolean_col1": 0.08, # 8%概率为空
    "string_col2": 0.10, # 10%概率为空
    "boolean_col2": 0.08 # 8%概率为空
}

# 生成示例数据，随机添加空值
data = []
for i in range(1000):
    # 生成各列值，根据概率决定是否设为空
    id_val = i
    feature1_val = random.normalvariate(0, 1) if random.random() > null_probabilities["feature1"] else None
    feature2_val = random.normalvariate(0, 1) if random.random() > null_probabilities["feature2"] else None
    category_val = random.randint(1, 5) if random.random() > null_probabilities["category"] else None
    string_val1 = random.choice(categories) if random.random() > null_probabilities["string_col1"] else None
    boolean_val1 = random.choice([True, False]) if random.random() > null_probabilities["string_col2"] else None
    string_val2 = random.choice(categories) if random.random() > null_probabilities["string_col2"] else None
    boolean_val2 = random.choice([True, False]) if random.random() > null_probabilities["boolean_col2"] else None

    data.append((id_val, feature1_val, feature2_val, category_val, string_val1, boolean_val1, string_val2, boolean_val2))

# 定义Schema，明确各列类型
schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("feature1", DoubleType(), True),
    StructField("feature2", DoubleType(), True),
    StructField("category", IntegerType(), True),
    StructField("string_col1", StringType(), True),
    StructField("boolean_col1", BooleanType(), True),
    StructField("string_col2", StringType(), True),
    StructField("boolean_col2", BooleanType(), True)
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

print("\n=== 4. 变量与label的相关性和卡方检验 ===")
numeric_cols = [c for c, t in df.dtypes if t in ['int', 'double'] and c not in ['id']]

print("\n--- 连续变量与label的皮尔逊相关性 ---")
for col_name in numeric_cols:
    corr_value = df.stat.corr(col_name, "label")
    print(f"{col_name} 与 label 的皮尔逊相关系数: {corr_value:.4f}")

print("\n--- 分类变量与label的卡方检验 ---")
# 获取所有分类列（字符串列、布尔列和category列）
string_cols = [c for c, t in df.dtypes if t == 'string']
bool_cols = [c for c, t in df.dtypes if t == 'boolean']
cat_cols = string_cols + bool_cols + ['category']

for col_name in cat_cols:
    # 处理布尔型和数值型分类变量
    if col_name in bool_cols or col_name == 'category':
        encoded_col = f"{col_name}_int"
        if col_name in bool_cols:
            df = df.withColumn(encoded_col, col(col_name).cast("integer"))
        else:
            encoded_col = col_name  # category列已经是整数

        # 创建干净的DataFrame用于卡方检验
        chi_df = df.select(encoded_col, "label").dropna()

        # 转换为向量格式
        assembler = VectorAssembler(
            inputCols=[encoded_col],
            outputCol=f"{col_name}_vector",
            handleInvalid="skip"
        )
        vector_df = assembler.transform(chi_df)

        # 执行卡方检验
        chi_result = ChiSquareTest.test(vector_df, f"{col_name}_vector", "label").head()
        p_value = chi_result.pValues[0] if hasattr(chi_result, "pValues") else chi_result.pValue
        print(f"{col_name} 与 label 的卡方检验 p值: {p_value:.4f}")

    # 处理字符串型分类变量
    else:
        indexed_col = f"{col_name}_indexed"

        # 检查索引列是否已存在
        if indexed_col not in df.columns:
            indexer = StringIndexer(inputCol=col_name, outputCol=indexed_col, handleInvalid='skip')
            df = indexer.fit(df).transform(df)

        # 创建干净的DataFrame用于卡方检验
        chi_df = df.select(indexed_col, "label").dropna()

        # 转换为向量格式
        assembler = VectorAssembler(
            inputCols=[indexed_col],
            outputCol=f"{col_name}_vector",
            handleInvalid="skip"
        )
        vector_df = assembler.transform(chi_df)

        # 执行卡方检验
        chi_result = ChiSquareTest.test(vector_df, f"{col_name}_vector", "label").head()
        p_value = chi_result.pValues[0] if hasattr(chi_result, "pValues") else chi_result.pValue
        print(f"{col_name} 与 label 的卡方检验 p值: {p_value:.4f}")

# 保留原有的相关性矩阵可视化
print("\n--- 变量间相关性矩阵可视化 ---")
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
for col_name in string_cols:
    indexed_col = f"{col_name}_indexed"

    # 检查索引列是否已存在
    if indexed_col not in df.columns:
        indexer = StringIndexer(inputCol=col_name, outputCol=indexed_col, handleInvalid='skip')
        df = indexer.fit(df).transform(df)

    # 计算与label的相关性（使用编码后的值）
    corr_value = df.stat.corr(indexed_col, "label")
    print(f"{col_name} 与 label 的相关性: {corr_value:.4f}")

    # 创建干净的DataFrame用于卡方检验
    chi_df = df.select(indexed_col, "label").dropna()

    # 转换为向量格式
    assembler = VectorAssembler(
        inputCols=[indexed_col],
        outputCol=f"{col_name}_vector",
        handleInvalid="skip"
    )
    vector_df = assembler.transform(chi_df)

    # 卡方检验
    chi_result = ChiSquareTest.test(vector_df, f"{col_name}_vector", "label").head()
    p_value = chi_result.pValues[0] if hasattr(chi_result, "pValues") else chi_result.pValue
    print(f"{col_name} 与 label 的卡方检验 p值: {p_value:.4f}")

print("\n=== 6. 布尔型特征与标签的相关性 ===")
# 布尔型特征处理
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
    p_value = chi_result.pValues[0] if hasattr(chi_result, "pValues") else chi_result.pValue
    print(f"{col_name} 与 label 的卡方检验 p值: {p_value:.4f}")

print("\n=== 7. 类别型特征的分布分析 ===")
# 类别型特征分布
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