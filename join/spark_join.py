from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id, col, when

# 创建SparkSession
spark = SparkSession.builder.appName("MergeDerivedDataFrames").getOrCreate()

# 1. 创建原始DataFrame df1，并添加唯一标识列uid
data1 = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
df1 = spark.createDataFrame(data1, ["name", "age"]) \
    .withColumn("uid", monotonically_increasing_id())  # 添加唯一标识

# 2. 基于df1生成df2，保留uid列（已修复when条件的错误）
df2 = df1.select(
    col("uid"),  # 保留唯一标识
    col("name"),
    (col("age") * 100).alias("salary"),
    # 修复：将 0 改为 col("age") % 2 == 0 列表达式
    when(col("age") % 2 == 0, "HR").otherwise("Engineering").alias("department")
)

# 3. 合并df1和df2
merged_df = df1.join(
    df2.drop("name"),  # 移除重复列
    on="uid",
    how="inner"
).drop("uid")

# 展示结果
print("原始df1（含uid）：")
df1.show()
print("\n由df1生成的df2（含uid）：")
df2.show()
print("\n合并后的DataFrame：")
merged_df.show()
