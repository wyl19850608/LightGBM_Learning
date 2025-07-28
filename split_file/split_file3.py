import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id

def hdfs_split_by_line(input_path, output_dir, num_files):
    """
    在HDFS上按行分割文件，直接输出到指定目录，不创建子文件夹
    """
    # 初始化Spark会话
    spark = SparkSession.builder \
        .appName("HDFSLineSplitterFixedPath") \
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

        # 添加临时行号（用于筛选行范围）
        df = df.withColumn("temp_id", monotonically_increasing_id())

        # 按行号范围筛选并直接写入到输出目录
        for file_id in range(num_files):
            # 计算当前文件的行号范围
            start_id = file_id * lines_per_file
            # 最后一个文件处理剩余所有行
            end_id = (file_id + 1) * lines_per_file if file_id < num_files - 1 else total_lines

            # 筛选当前范围内的行，只保留原始文本列
            filtered = df.filter((df["temp_id"] >= start_id) & (df["temp_id"] < end_id)).select("value")

            # 修复：在text()方法中显式指定路径，确保文件名正确
            output_path = f"{output_dir}/part-{file_id:04d}.txt"
            filtered.coalesce(1).write \
                .mode("overwrite") \
                .text(output_path)

            print(f"已生成文件: {output_path}")

    finally:
        spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在HDFS上按行分割文件，直接输出到指定目录")
    parser.add_argument("--input", required=True, help="HDFS输入文件路径（如hdfs:///user/data/source.log）")
    parser.add_argument("--output", required=True, help="HDFS输出目录路径（如hdfs:///user/data/split_results）")
    parser.add_argument("--num-files", type=int, required=True, help="要拆分的文件数量")

    args = parser.parse_args()

    hdfs_split_by_line(
        input_path=args.input,
        output_dir=args.output,
        num_files=args.num_files
    )
    print(f"所有文件已拆分完成，直接保存至: {args.output}")
