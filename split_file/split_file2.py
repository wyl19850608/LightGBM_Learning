import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id

def hdfs_split_by_line(input_path, output_dir, num_files):
    """
    在HDFS上按行分割文件，不解析内容，确保每行完整完整
    """
    # 初始化Spark会话
    spark = SparkSession.builder \
        .appName("HDFSLineSplitter") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "16g") \
        .config("spark.executor.cores", "4") \
        .getOrCreate()

    try:
        # 以纯文本格式读取HDFS文件，每行作为一条记录
        df = spark.read.text(input_path)

        # 获取总行数
        total_lines = df.count()
        print(f"HDFS文件总行数: {total_lines}")
        lines_per_file = total_lines // num_files
        print(f"每个输出文件大约包含 {lines_per_file} 行")

        # 添加临时行号（后续会删除）
        df = df.withColumn("temp_id", monotonically_increasing_id())

        # 按行号范围筛选并写入，彻底避开分区列数据类型问题
        for file_id in range(num_files):
            # 计算当前文件的行号范围
            start_id = file_id * lines_per_file
            end_id = (file_id + 1) * lines_per_file if file_id < num_files - 1 else total_lines

            # 筛选当前范围内的行，只保留原始文本列
            filtered = df.filter((df["temp_id"] >= start_id) & (df["temp_id"] < end_id)).select("value")

            # 写入当前文件（合并为单个文件）
            filtered.coalesce(1).write \
                .mode("overwrite") \
                .text(f"{output_dir}/split_{file_id:04d}")

            print(f"已生成文件: split_{file_id:04d}")

    finally:
        spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在HDFS上按行分割文件，不解析内容")
    parser.add_argument("--input", required=True, help="HDFS输入文件路径（如hdfs:///user/data/source.log）")
    parser.add_argument("--output", required=True, help="HDFS输出目录路径（如hdfs:///user/data/split_results）")
    parser.add_argument("--num-files", type=int, required=True, help="要拆分的文件数量")

    args = parser.parse_args()

    hdfs_split_by_line(
        input_path=args.input,
        output_dir=args.output,
        num_files=args.num_files
    )
