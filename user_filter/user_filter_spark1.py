from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql import functions as F

def filter_dataframe_with_time(spark_df):
    # 按user_id和time排序
    window_spec = Window.partitionBy("user_id").orderBy("time")

    # 添加行号，每组内按时间排序
    ranked_df = spark_df.withColumn("row_num", F.row_number().over(window_spec))

    # 找出每组中label=1的记录，如果没有则取所有记录
    label_1_df = spark_df.filter(F.col("label") == 1)
    has_label_1 = label_1_df.groupBy("user_id").agg(F.count("*").alias("has_label_1"))

    # 合并信息
    merged_df = spark_df.join(has_label_1, "user_id", "left")

    # 筛选条件：如果有label=1的记录，则取label=1且时间最晚的记录；否则取时间最晚的记录
    result_df = merged_df.filter(
        ((F.col("has_label_1") >= 1) & (F.col("label") == 1)) |
        (F.col("has_label_1").isNull())
    )

    # 再次使用窗口函数获取每组中时间最晚的记录
    latest_window = Window.partitionBy("user_id").orderBy(F.desc("time"))
    final_df = result_df.withColumn("rank", F.rank().over(latest_window)) \
        .filter(F.col("rank") == 1) \
        .drop("rank", "has_label_1")

    return final_df

# 示例用法
if __name__ == "__main__":
    spark = SparkSession.builder.appName("FilterDataFrame").getOrCreate()

    # 创建带时间字段的示例数据
    data = [
        (1, 0, '2023-01-01', 'a'),
        (1, 1, '2023-01-02', 'b'),
        (1, 0, '2023-01-03', 'c'),
        (2, 0, '2023-01-01', 'd'),
        (2, 0, '2023-01-02', 'e'),
        (3, 1, '2023-01-01', 'f'),
        (3, 1, '2023-01-02', 'g'),
        (3, 0, '2023-01-03', 'h'),
        (4, 0, '2023-01-01', 'i'),
        (4, 1, '2023-01-02', 'j')
    ]

    columns = ["user_id", "label", "time", "other_data"]
    df = spark.createDataFrame(data, columns)

    # 将时间字符串转换为时间戳
    df = df.withColumn("time", F.to_timestamp("time", "yyyy-MM-dd"))

    # 调用函数进行过滤
    filtered_df = filter_dataframe_with_time(df)

    print("原始数据:")
    df.show()
    print("\n过滤后的数据:")
    filtered_df.show()

    spark.stop()