from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StructType, StructField, IntegerType
import numpy as np

def process_days(col):
    # 处理NULL值
    if col == "NULL" or col is None:
        return (None, None, None)

    try:
        # 去除空格并转换为整数列表
        days = [int(x.strip()) for x in col.split(',') if x.strip()]

        # 计算统计量
        min_val = min(days)
        mean_val = round(np.mean(days))
        max_val = max(days)

        return (min_val, mean_val, max_val)
    except (ValueError, AttributeError):
        return (None, None, None)

if __name__ == "__main__":
    # 初始化SparkSession
    spark = SparkSession.builder \
        .appName("DaysProcessing") \
        .getOrCreate()

    # 定义返回数据类型
    output_schema = StructType([
        StructField("min", IntegerType(), nullable=True),
        StructField("mean", IntegerType(), nullable=True),
        StructField("max", IntegerType(), nullable=True)
    ])

    # 创建UDF
    process_days_udf = udf(process_days, output_schema)

    # 创建测试数据
    data = [
        ('3, 5, 7, 9',),       # 正常的逗号分隔数字
        (' 10 , 20 , 30 ',),   # 带空格的情况
        ('NULL',),             # NULL字符串
        (None,),               # None值
        ('abc, 123',),         # 包含非数字的情况
        ('8',)                 # 单个数字
    ]

    # 创建DataFrame
    df = spark.createDataFrame(data, ["days_column"])

    # 应用UDF处理数据
    result_df = df.withColumn(
        "stats",
        process_days_udf(col("days_column"))
    ).select(
        "days_column",
        col("stats.min"),
        col("stats.mean"),
        col("stats.max")
    )

    # 显示结果
    print("处理前的数据:")
    df.show(truncate=False)

    print("\n处理后的结果:")
    result_df.show(truncate=False)

    # 停止SparkSession
    spark.stop()
