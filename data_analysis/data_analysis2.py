import os
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan, when, count, rand, corr, skewness, kurtosis
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.stat import Correlation, ChiSquareTest
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
data = [
    (
        i,                          # id: 整数
        random.normalvariate(0, 1),  # feature1: 浮点数（Python float）
        random.normalvariate(0, 1),  # feature2: 浮点数（Python float）
        random.randint(1, 5),        # category: 整数
        random.choice(categories),  # string_col: 字符串
        random.choice([True, False]) # boolean_col: 布尔值
    )
    for i in range(100)
]

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
df.show(100, truncate=False)

# 以下代码与之前版本类似，但优化了相关性矩阵可视化部分
print("\n=== 4. 相关性矩阵可视化 ===")
# 筛选数值型列（排除id和label）
numeric_cols = [c for c, t in df.dtypes if t in ['int', 'double'] and c not in ['id', 'label']]

print("numeric_cols:", numeric_cols)

# 转换为MLlib向量格式
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
print("assembler:", assembler)
vector_df = assembler.transform(df).select("features")
print("vector_df:", vector_df)

# 计算相关矩阵
corr_matrix = Correlation.corr(vector_df, "features").collect()[0][0].toArray()

# 使用Pandas和Matplotlib可视化（确保不传递NumPy数组）
plt.figure(figsize=(10, 8))
corr_df = pd.DataFrame(corr_matrix, columns=numeric_cols, index=numeric_cols)
plt.imshow(corr_df.values, cmap='coolwarm', interpolation='nearest')  # 显式使用pd.DataFrame.values
plt.colorbar()
plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=45)
plt.yticks(range(len(corr_df.columns)), corr_df.columns)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
print("相关性矩阵已保存为: correlation_matrix.png")

# 其余代码（缺失值分析、统计量计算等）保持不变，仅确保不使用NumPy

# 停止SparkSession
spark.stop()
print("\n=== 分析完成 ===")