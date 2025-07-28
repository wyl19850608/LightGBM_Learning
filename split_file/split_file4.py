import argparse
from pyspark.sql import SparkSession

def split_file(input_path, output_dir, num_files):
    # 初始化Spark会话
    spark = SparkSession.builder \
        .appName("LineSplitter") \
        .getOrCreate()

    try:
        # 读取文件（每行作为一条完整记录）
        rdd = spark.sparkContext.textFile(input_path)

        # 计算总记录数
        total = rdd.count()
        print(f"总记录数: {total}")

        # 按数量均匀分割RDD
        split_rdds = rdd.randomSplit([1.0/num_files]*num_files)

        # 逐个个分割后的结果写入指定目录
        for i, rdd_part in enumerate(split_rdds):
            # 直接写入文件，使用coalesce(1)确保每个分区一个文件
            rdd_part.coalesce(1).saveAsTextFile(f"{output_dir}/temp_{i}")

            # 使用HDFS命令移动文件（避免子文件夹）
            import subprocess
            # 找到临时生成的文件
            find_cmd = f"hdfs dfs -ls {output_dir}/temp_{i} | grep -v _SUCCESS | awk '{{print $8}}'"
            temp_file = subprocess.check_output(find_cmd, shell=True).decode().strip()

            # 移动到目标目录并重命名
            target_file = f"{output_dir}/part_{i:04d}.txt"
            subprocess.run(f"hdfs dfs -mv {temp_file} {target_file}", shell=True, check=True)

            # 清理临时目录
            subprocess.run(f"hdfs dfs -rm -r {output_dir}/temp_{i}", shell=True, check=True)

            print(f"生成文件: {target_file}")

    finally:
        spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入文件路径（HDFS或本地）")
    parser.add_argument("--output", required=True, help="输出目录（HDFS或本地）")
    parser.add_argument("--num-files", type=int, required=True, help="分割文件数量")
    args = parser.parse_args()

    split_file(args.input, args.output, args.num_files)
