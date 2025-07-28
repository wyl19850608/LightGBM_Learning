from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

def user_filter(df: DataFrame) -> DataFrame:
    """
    对用户数据进行过滤处理，按user_id分组：
    - 若组内存在label为1的记录，保留最后一条label为1的记录
    - 若组内不存在label为1的记录，保留组内最后一条记录

    参数:
        df: 包含user_id, label和其他数据列的DataFrame

    返回:
        过滤后的DataFrame，保留原始所有列
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
        # 标记组内是否有label=1的记录
        F.max(F.when(F.col("label") == 1, 1).otherwise(0)).alias("has_one"),
        # 组内最后一条label=1记录的行号
        F.max(F.when(F.col("label") == 1, F.col("rn")).otherwise(None)).alias("max_one_rn"),
        # 组内最后一条记录的行号
        F.max("rn").alias("max_rn")
    )

    # 步骤3：确定每个user_id需要保留的记录行号
    selected_rn = grouped.withColumn(
        "selected_rn",
        # 如果有label=1的记录，选择最后一条label=1的记录，否则选择最后一条记录
        F.when(F.col("has_one") == 1, F.col("max_one_rn")).otherwise(F.col("max_rn"))
    ).select("user_id", "selected_rn")

    # 步骤4：为DataFrame设置别名，解决连接后的列名歧义问题
    df_with_rn_alias = df_with_rn.alias("left")
    selected_rn_alias = selected_rn.alias("right")

    # 关联原始数据，获取最终结果（选择左表所有列）
    result = df_with_rn_alias.join(
        selected_rn_alias,
        (df_with_rn_alias.user_id == selected_rn_alias.user_id) &
        (df_with_rn_alias.rn == selected_rn_alias.selected_rn),
        how="inner"
    ).select(df_with_rn_alias["*"]) \
        .drop("rn")  # 删除临时添加的行号列

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
        .appName("用户过滤示例") \
        .getOrCreate()

    # 定义示例数据（增加一列演示多列保留效果）
    data = [
        (1, 0, "a", "extra1"),
        (1, 1, "b", "extra2"),
        (1, 0, "c", "extra3"),
        (2, 0, "d", "extra4"),
        (2, 0, "e", "extra5"),
        (3, 1, "f", "extra6"),
        (3, 1, "g", "extra7"),
        (3, 0, "h", "extra8"),
        (4, 0, "i", "extra9"),
        (4, 1, "j", "extra10")
    ]

    # 创建DataFrame（包含额外列）
    df = spark.createDataFrame(data, ["user_id", "label", "other_data", "additional_col"])

    # 调用用户过滤函数
    filtered_result = user_filter(df)

    # 停止SparkSession
    spark.stop()
