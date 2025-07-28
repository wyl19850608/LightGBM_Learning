from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import pandas as pd

def dict_to_hdfs(data_dict, hdfs_path, file_format="parquet", partition_cols=None):
    """
    将Python字典转换为DataFrame并存储到HDFS

    参数:
    data_dict (dict): 要转换的Python字典
    hdfs_path (str): HDFS存储路径
    file_format (str): 文件格式，支持'parquet'、'csv'、'json'等
    partition_cols (list): 分区列名列表
    """
    try:
        # 创建SparkSession
        spark = SparkSession.builder \
            .appName("DictToHDFS") \
            .getOrCreate()

        # 示例：定义DataFrame结构（根据实际数据调整）
        schema = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
            StructField("city", StringType(), True)
        ])

        # 将Python字典转换为Pandas DataFrame
        pandas_df = pd.DataFrame(data_dict)

        # 将Pandas DataFrame转换为Spark DataFrame
        spark_df = spark.createDataFrame(pandas_df, schema=schema)

        # 打印DataFrame结构和前几行
        spark_df.printSchema()
        spark_df.show(5)

        # 写入HDFS
        writer = spark_df.write

        # 如果指定了分区列，则进行分区
        if partition_cols:
            writer = writer.partitionBy(*partition_cols)

        # 根据指定格式写入HDFS
        if file_format == "parquet":
            writer.parquet(hdfs_path, mode="overwrite")
        elif file_format == "csv":
            writer.csv(hdfs_path, mode="overwrite", header=True)
        elif file_format == "json":
            writer.json(hdfs_path, mode="overwrite")
        else:
            print(f"不支持的文件格式: {file_format}")
            return False

        print(f"数据已成功写入 HDFS: {hdfs_path}")
        return True

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        return False
    finally:
        # 关闭SparkSession
        if 'spark' in locals():
            spark.stop()

# 示例用法
if __name__ == "__main__":
    # 示例字典数据
    sample_data = {
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "age": [25, 30, 35, 40, 45],
        "city": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
    }

    # HDFS存储路径
    hdfs_output_path = "hdfs://localhost:9000/user/data/output/sample_data"

    # 转换并存储数据
    dict_to_hdfs(sample_data, hdfs_output_path, file_format="parquet")