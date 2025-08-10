from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import lit, col
from pyspark.sql.types import StringType, IntegerType, FloatType
import numpy as np
from typing import List, Tuple

def sample_sampled_dataframe(
        source_df: DataFrame,
        source_cols: List[str],
        target_cols: List[str],
        sample_ratio: float = None,
        sample_count: int = None
) -> Tuple[DataFrame, List[str]]:
    """
    处理无列名的源DataFrame，先使用source_cols命名列，再添加目标列并抽样，确保输出列顺序与target_cols一致

    参数:
        source_df: 无列名的源Spark DataFrame
        source_cols: 用于为源DataFrame命名的列名数组（需与源DataFrame列数一致）
        target_cols: 目标列名数组（source_cols全部包含在其中）
        sample_ratio: 抽样比例（0到1之间）
        sample_count: 抽样条数

    返回:
        抽样后的Spark DataFrame（列顺序与target_cols一致）和目标列名数组
    """
    # 验证输入参数
    if sample_ratio is None and sample_count is None:
        raise ValueError("必须指定抽样比例(sample_ratio)或抽样条数(sample_count)")

    if sample_ratio is not None:
        if not (0 < sample_ratio <= 1):
            raise ValueError("抽样比例必须在0到1之间（不包括0）")

    if sample_count is not None:
        if not (isinstance(sample_count, int) and sample_count > 0):
            raise ValueError("抽样条数必须是正整数")

    # 验证源DataFrame列数与source_cols长度一致
    source_df_col_count = len(source_df.columns)
    if len(source_cols) != source_df_col_count:
        raise ValueError(f"必须与源DataFrame列数一致")

    # 验证source_cols全部存在于target_cols中
    for col_name in source_cols:
        if col_name not in target_cols:
            raise ValueError(f"source_cols中的列不存在于target_cols中")

    # 为源DataFrame添加列名（按顺序对应）
    renamed_df = source_df.toDF(*source_cols)

    # 添加目标列中不存在于源列的列（初始值为空值）
    result_df = renamed_df
    for i, tgt_col in enumerate(target_cols):
        if tgt_col in result_df.columns:
            continue
        else:
            # 确定列的数据类型
            # if i < len(source_cols):
            #     src_col = source_cols[i]
            #     col_type = result_df.schema[src_col].dataType
            # else:
            col_type = StringType()

            # 添加新的目标列，值为空值
            result_df = result_df.withColumn(tgt_col, lit(None).cast(col_type))

    # 确保列顺序与target_cols完全一致
    result_df = result_df.select(target_cols)

    # 执行抽样
    total_count = result_df.count()
    ratio = sample_ratio
    if sample_count is not None:
        ratio = min(1.0, sample_count / total_count)
    ratio = min(1.0, ratio*1.2)
    temp_sampled = result_df.sample(withReplacement=False, fraction=ratio, seed=np.random.randint(0, 10000))
    if sample_ratio is not None:
        sample_count = min(total_count, int(sample_ratio * total_count))
    sampled_df = temp_sampled.limit(sample_count)
    sampled_df.show()

    return sampled_df, target_cols


def test_sample_sampled_dataframe():
    """测试sample_sampled_dataframe函数的各种场景"""
    # 初始化SparkSession
    spark = SparkSession.builder \
        .appName("SampleDataFrameTest") \
        .master("local[*]") \
        .getOrCreate()

    try:
        # 1. 创建测试数据 - 无列名的源DataFrame
        data = [
            (1, "Alice", 25.5),
            (2, "Bob", 30.2),
            (3, "Charlie", 35.7),
            (4, "David", 40.1),
            (5, "Eve", 28.9),
            (6, "Frank", 33.3),
            (7, "Grace", 45.6),
            (8, "Heidi", 29.8),
            (9, "Ivan", 31.2),
            (10, "Judy", 37.4)
        ]
        # 创建无列名的DataFrame（默认列名为_c0, _c1, _c2）
        source_df = spark.createDataFrame(data)

        # 测试用的源列名和目标列名
        source_cols = ["id", "name", "score"]
        target_cols = ["id", "name", "score", "age", "gender", "address"]

        print("测试1: 使用抽样比例...")
        sampled_df1, tgt_cols1 = sample_sampled_dataframe(
            source_df=source_df,
            source_cols=source_cols,
            target_cols=target_cols,
            sample_ratio=0.5
        )

        # 验证结果
        assert tgt_cols1 == target_cols, "测试1: 目标列名不匹配"
        assert sampled_df1.columns == target_cols, "测试1: 列顺序不正确"
        assert len(sampled_df1.columns) == len(target_cols), "测试1: 列数量不正确"
        assert 3 <= sampled_df1.count() <= 7, f"测试1: 抽样数量不正确，实际为{sampled_df1.count()}"
        print("测试1: 成功")

        print("\n测试2: 使用抽样条数...")
        sampled_df2, tgt_cols2 = sample_sampled_dataframe(
            source_df=source_df,
            source_cols=source_cols,
            target_cols=target_cols,
            sample_count=3
        )

        # 验证结果
        assert tgt_cols2 == target_cols, "测试2: 目标列名不匹配"
        assert sampled_df2.columns == target_cols, "测试2: 列顺序不正确"
        assert sampled_df2.count() == 3, f"测试2: 抽样数量不正确，实际为{sampled_df2.count()}"
        print("测试2: 成功")

        print("\n测试3: 抽样条数大于总条数...")
        sampled_df3, tgt_cols3 = sample_sampled_dataframe(
            source_df=source_df,
            source_cols=source_cols,
            target_cols=target_cols,
            sample_count=20
        )

        # 验证结果
        assert sampled_df3.count() == 10, f"测试3: 应返回全部数据，实际为{sampled_df3.count()}"
        print("测试3: 成功")

        print("\n测试4: 目标列与源列数量相同...")
        target_cols4 = ["id", "name", "score"]
        sampled_df4, tgt_cols4 = sample_sampled_dataframe(
            source_df=source_df,
            source_cols=source_cols,
            target_cols=target_cols4,
            sample_count=5
        )

        assert sampled_df4.columns == target_cols4, "测试4: 列顺序不正确"
        assert sampled_df4.count() == 5, f"测试4: 抽样数量不正确，实际为{sampled_df4.count()}"
        print("测试4: 成功")

        print("\n测试5: 验证异常处理 - 未指定抽样参数...")
        try:
            sample_sampled_dataframe(
                source_df=source_df,
                source_cols=source_cols,
                target_cols=target_cols
            )
            assert False, "测试5: 应抛出异常但未抛出"
        except ValueError as e:
            assert "必须指定抽样比例" in str(e), "测试5: 异常信息不正确"
        print("测试5: 成功")

        print("\n测试6: 验证异常处理 - source_cols与源DataFrame列数不匹配...")
        try:
            sample_sampled_dataframe(
                source_df=source_df,
                source_cols=["id", "name"],  # 长度不匹配
                target_cols=target_cols,
                sample_count=3
            )
            assert False, "测试6: 应抛出异常但未抛出"
        except ValueError as e:
            assert "必须与源DataFrame列数一致" in str(e), "测试6: 异常信息不正确"
            if "必须与源DataFrame列数一致" in str(e):
                print("测试6: 成功")
            else:
                assert "source_cols" in str(e), "测试6: 异常信息不正确"

        print("\n测试7: 验证异常处理 - source_cols不在target_cols中...")
        try:
            sample_sampled_dataframe(
                source_df=source_df,
                source_cols=["id", "name", "invalid_col"],  # 包含不在target_cols中的列
                target_cols=target_cols,
                sample_count=3
            )
            assert False, "测试7: 应抛出异常但未抛出"
        except ValueError as e:
            assert "不存在于target_cols中" in str(e), "测试7: 异常信息不正确"
        print("测试7: 成功")

        print("\n所有测试通过!")

    except AssertionError as ae:
        print(f"测试失败: {str(ae)}")
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")
    finally:
        spark.stop()


if __name__ == "__main__":
    test_sample_sampled_dataframe()
