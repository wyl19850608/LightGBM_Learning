import option
from pyspark.sql import SparkSession
import argparse


def split_large_csv(input_path, output_path, num_files, has_header=True):
    """
    将HDFS上的大型CSV文件拆分为指定数量的小文件，确保行记录完整

    参数:
    input_path: HDFS上输入CSV文件的路径
    output_path: HDFS上输出拆分文件的目录
    num_files: 要拆分的文件数量
    has_header: 是否包含表头
    """
    # 初始化SparkSession
    spark = SparkSession.builder \
        .appName("SplitLargeCSV") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "16g") \
        .config("spark.executor.cores", "4") \
        .config("spark.sql.shuffle.partitions", num_files) \
        .getOrCreate()

    try:
        # 读取CSV文件，确保按行解析
        df = spark.read \
            .option("header", has_header) \
            .option("inferSchema", "false")  \
            .option("multiLine", "false")  \
            .option("quote", "\"") \
            .option("escape", "\"") \
            .csv(input_path)

        # 获取总行数以计算每个文件的大致行数
        total_rows = df.count()
        print(f"总记录数: {total_rows}")
        rows_per_file = total_rows // num_files
        print(f"每个文件大约包含 {rows_per_file} 行记录")

        # 如果有表头，单独处理表头
        if has_header:
            # 获取表头
            header = df.columns
            # 去除表头行
            data_rows = df.rdd.zipWithIndex().filter(lambda x: x[1] > 0).map(lambda x: x[0])
            df = spark.createDataFrame(data_rows, schema=df.schema)

        # 按指定数量的文件拆分并保存
        (df.repartition(num_files)
         .write
         .option("header", has_header)
         .option("quote", "\"")
         .option("escape", "\"")
         .mode("overwrite")
         .csv(output_path))

        print(f"成功将文件拆分为 {num_files} 个文件，存储在 {output_path}")

    finally:
    # 关闭SparkSession
        spark.stop()

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='拆分HDFS上的大型CSV文件')
    parser.add_argument('--input', required=True, help='HDFS上输入CSV文件的路径')
    parser.add_argument('--output', required=True, help='HDFS上输出拆分文件的目录')
    parser.add_argument('--num-files', type=int, default=10, help='要拆分的文件数量')
    parser.add_argument('--has-header', action='store_true', help='输入文件是否包含表头')

    args = parser.parse_args()

    # 调用拆分函数
    split_large_csv(
        input_path=args.input,
        output_path=args.output,
        num_files=args.num_files,
        has_header=args.has_header
    )
