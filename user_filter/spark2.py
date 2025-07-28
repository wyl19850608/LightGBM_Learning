from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

def process_user_data(df: DataFrame) -> DataFrame:
    """
    处理用户数据，按user_id分组：
    - 如果组内存在label为1的记录，保留最后一条label为1的记录
    - 否则保留最后组内最后一条记录

    参数:
        df: 包含user_id, label和其他数据的DataFrame

    返回:
        处理后的DataFrame
    """
    # 打印处理前的数据
    print("="*50)
    print("处理前的数据:")
    df.show()
    print("="*50)

    # 步骤1：添加行号以保持原始数据顺序
    df_with_rn = df.withColumn("rn", F.monotonically_increasing_id())

    # 步骤2：按user_id分组，计算关键值
    grouped = df_with_rn.groupBy("user_id").agg(
        F.max(F.when(F.col("label") == 1, 1).otherwise(0)).alias("has_one"),
        F.max(F.when(F.col("label") == 1, F.col("rn")).otherwise(None)).alias("max_one_rn"),
        F.max("rn").alias("max_rn")
    )

    # 步骤3：确定每个user_id需要保留的记录行号
    selected_rn = grouped.withColumn(
        "selected_rn",
        F.when(F.col("has_one") == 1, F.col("max_one_rn")).otherwise(F.col("max_rn"))
    ).select("user_id", "selected_rn")

    # 步骤4：为DataFrame设置别名，解决连接后的列名歧义问题
    df_with_rn_alias = df_with_rn.alias("left")
    selected_rn_alias = selected_rn.alias("right")

    # 关联原始数据，获取最终结果
    result = df_with_rn_alias.join(
        selected_rn_alias,
        (df_with_rn_alias.user_id == selected_rn_alias.user_id) &
        (df_with_rn_alias.rn == selected_rn_alias.selected_rn),
        how="inner"
    ).select(
        df_with_rn_alias.user_id,
        df_with_rn_alias.label,
        df_with_rn_alias.other_data
    )

    # 打印处理后的数据
    print("\n" + "="*50)
    print("处理后的数据:")
    result.show()
    print("="*50)

    return result

# 示例用法
if __name__ == "__main__":
    # 初始化SparkSession
    spark = SparkSession.builder \
        .appName("UserdataProcessor") \
        .getOrCreate()

    # 定义原始数据
    data = [
        (1, 0, "a"),
        (1, 1, "b"),
        (1, 0, "c"),
        (2, 0, "d"),
        (2, 0, "e"),
        (3, 1, "f"),
        (3, 1, "g"),
        (3, 0, "h"),
        (4, 0, "i"),
        (4, 1, "j")
    ]

    # 创建DataFrame
    df = spark.createDataFrame(data, ["user_id", "label", "other_data"])

    # 调用处理函数
    processed_df = process_user_data(df)

    # 停止SparkSession
    spark.stop()
