import os
os.environ["PYSPARK_PYTHON"] = "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10"

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan, when, count, rand, corr, skewness, kurtosis
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType, BooleanType


# 创建SparkSession
spark = SparkSession.builder \
    .appName("DataAnalysis") \
    .getOrCreate()

# 设置日志级别以减少警告信息
spark.sparkContext.setLogLevel("ERROR")

# 示例：读取数据集（请替换为实际数据路径）
# df = spark.read.csv("path/to/your/data.csv", header=True, inferSchema=True)

# 生成示例数据用于演示
# 定义可能的字符串类别
data = [(i,
         np.random.randn(),
         np.random.randn(),
         np.random.randint(1, 5),
         ) for i in range(10)]

columns = ["id", "feature1", "feature2", "category"]

df = spark.createDataFrame(data, columns)

# 1. 添加随机label列(0/1)
df = df.withColumn("label", (rand() > 0.5).cast("integer"))

# 2. 缺失值统计
missing_counts = df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns])
missing_pct = missing_counts.select([(col(c) / df.count() * 100).alias(f"{c}_missing_pct") for c in missing_counts.columns])
missing_pct.show()

# 3. 数据分布分析（数值型特征）
numeric_cols = [col_type[0] for col_type in df.dtypes if col_type[1] in ['int', 'double'] and col_type[0] != 'id']
distribution_stats = df.select(numeric_cols).describe()
distribution_stats.show()

# 计算偏度和峰度
skewness_kurtosis = df.select([skewness(c).alias(f"{c}_skewness") for c in numeric_cols] +
                              [kurtosis(c).alias(f"{c}_kurtosis") for c in numeric_cols])
skewness_kurtosis.show()

# 4. Label与其他特征的相关性分析

# 4.1 数值型特征：Pearson相关系数
pearson_corr = {}
for col_name in numeric_cols:
    if col_name != 'label':
        corr_value = df.stat.corr('label', col_name, method='pearson')
        pearson_corr[col_name] = corr_value

# 4.2 分类型特征：使用卡方检验
from pyspark.ml.feature import StringIndexer
from pyspark.ml.stat import ChiSquareTest

categorical_cols = [col_type[0] for col_type in df.dtypes if col_type[1] in ['string', 'boolean']]
chi_square_results = {}

for col_name in categorical_cols:
    # 索引分类特征
    indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_indexed")
    indexed_df = indexer.fit(df).transform(df)

    # 卡方检验
    assembler = VectorAssembler(inputCols=[f"{col_name}_indexed"], outputCol="features")
    assembled_df = assembler.transform(indexed_df)

    chi_test = ChiSquareTest.test(assembled_df, "features", "label")
    chi_result = chi_test.head()
    chi_square_results[col_name] = {
        "pValue": chi_result.pValues[0],
        "statistic": chi_result.statistics[0]
    }

# 5. 相关性矩阵可视化（数值型特征）
# 转换为MLlib向量格式
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
vector_df = assembler.transform(df).select("features")

# 计算相关矩阵
corr_matrix = Correlation.corr(vector_df, "features").collect()[0][0].toArray()

# 使用Pandas和Matplotlib可视化
plt.figure(figsize=(10, 8))
corr_df = pd.DataFrame(corr_matrix, columns=numeric_cols, index=numeric_cols)
plt.imshow(corr_df, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=45)
plt.yticks(range(len(corr_df.columns)), corr_df.columns)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png")  # 在PySpark环境中保存图表

# 6. 输出结果
print("\n=== Pearson Correlation with Label ===")
for col, corr_val in pearson_corr.items():
    print(f"{col}: {corr_val:.4f}")

print("\n=== Chi-Square Test Results (Categorical Features) ===")
for col, result in chi_square_results.items():
    print(f"{col}: p-value={result['pValue']:.4f}, statistic={result['statistic']:.4f}")

# 7. 保存分析结果到CSV
missing_pct.toPandas().to_csv("missing_values.csv", index=False)
distribution_stats.toPandas().to_csv("distribution_stats.csv", index=False)
pd.DataFrame(pearson_corr.items(), columns=['Feature', 'PearsonCorrelation']).to_csv("pearson_correlation.csv", index=False)

# 停止SparkSession
spark.stop()