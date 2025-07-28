import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import StringIndexer
from pyspark.ml.stat import ChiSquareTest

# 设置PySpark环境变量
os.environ['PYSPARK_SUBMIT_ARGS'] = '--conf "spark.driver.extraJavaOptions=-XX:+IgnoreUnrecognizedVMOptions --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED" --conf "spark.executor.extraJavaOptions=-XX:+IgnoreUnrecognizedVMOptions --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED" pyspark-shell'
os.environ["PYSPARK_PYTHON"] = "/Users/wangyanling32/code/pythonProject4/.venv/bin/python3.8"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/Users/wangyanling32/code/pythonProject4/.venv/bin/python3.8"

# 创建SparkSession
spark = SparkSession.builder \
    .appName("DataAnalysis") \
    .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC --add-opens java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED") \
    .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC --add-opens java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED") \
    .getOrCreate()

# 设置日志级别以减少警告信息
spark.sparkContext.setLogLevel("ERROR")

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
    "string_col2": 0.10, # 10%概率为空
    "boolean_col1": 0.08, # 8%概率为空
    "boolean_col2": 0.08  # 8%概率为空
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
    string_val2 = random.choice(categories) if random.random() > null_probabilities["string_col2"] else None
    boolean_val1 = random.choice([True, False]) if random.random() > null_probabilities["boolean_col1"] else None
    boolean_val2 = random.choice([True, False]) if random.random() > null_probabilities["boolean_col2"] else None

    data.append((id_val, feature1_val, feature2_val, category_val, string_val1, string_val2, boolean_val1, boolean_val2))

from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType, BooleanType

# 定义Schema，明确各列类型
schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("feature1", DoubleType(), True),
    StructField("feature2", DoubleType(), True),
    StructField("category", IntegerType(), True),
    StructField("string_col1", StringType(), True),
    StructField("string_col2", StringType(), True),
    StructField("boolean_col1", BooleanType(), True),
    StructField("boolean_col2", BooleanType(), True)
])

# 创建DataFrame
df = spark.createDataFrame(data, schema)

# 添加随机label列(0/1)，使用Spark原生函数
df = df.withColumn("label", (F.rand() > 0.5).cast("integer"))

# 获取数值列
numeric_cols = [c for c, t in df.dtypes if t in ['int', 'double']]

# 删除包含任何null值的行
df = df.na.drop(subset=numeric_cols)

# 使用Spark内置方法计算相关性矩阵
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
df_vector = assembler.transform(df).select("features")
matrix = Correlation.corr(df_vector, "features").collect()[0][0]
corr_array = matrix.toArray()
corr_df = pd.DataFrame(corr_array, columns=numeric_cols, index=numeric_cols)

print("数值列相关性矩阵:")
print(corr_df)

# 保存数值列相关性矩阵到CSV文件
corr_df.to_csv("correlation_matrix_numeric.csv", index=True)
print("\n数值列相关性矩阵已保存到 'correlation_matrix_numeric.csv'")

# 获取字符串和布尔类型列
string_bool_cols = [c for c, t in df.dtypes if t in ['string', 'boolean']]

# 计算字符串和布尔类型列与label的卡方检验结果
chi_square_results = {}
for col in string_bool_cols:
    # 过滤掉包含空值的行
    valid_df = df.select(col, 'label').dropna()
    # 对字符串列进行索引化
    if valid_df.schema[col].dataType == StringType():
        indexer = StringIndexer(inputCol=col, outputCol=f"{col}_indexed")
        valid_df = indexer.fit(valid_df).transform(valid_df)
        input_col = f"{col}_indexed"
    else:
        input_col = col

    assembler = VectorAssembler(inputCols=[input_col], outputCol="features")
    assembled_df = assembler.transform(valid_df)
    chi_square_result = ChiSquareTest.test(assembled_df, "features", "label").head()
    chi_square_results[col] = {
        "pValue": chi_square_result.pValues[0],
        "statistic": chi_square_result.statistics[0]
    }

# 打印字符串和布尔类型列与label的卡方检验结果
print("\n字符串和布尔类型列与label的卡方检验结果:")
for col, result in chi_square_results.items():
    print(f"列名: {col}, p值: {result['pValue']}, 统计量: {result['statistic']}")

# 停止SparkSession
spark.stop()