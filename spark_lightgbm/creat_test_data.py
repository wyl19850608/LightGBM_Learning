# generate_test_data_csv.py
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
import os

def generate_test_data(spark, n_samples=5000):
    """生成测试数据（Spark DataFrame）"""
    np.random.seed(42)  # 设置随机种子，保证结果可复现

    # 1. 生成ID列
    data = {
        'user_id': np.random.randint(1000, 9999, size=n_samples),  # 用户ID（4位整数）
        'unique_id': [f'unique_{i}' for i in range(n_samples)]  # 唯一标识（字符串格式）
    }

    # 2. 生成分类特征（前5列）
    for i in range(5):
        # 每个分类特征从cat_0到cat_4中随机选择
        data[f'feat_{i}'] = np.random.choice([f'cat_{j}' for j in range(5)], size=n_samples)

    # 3. 生成数值特征（后15列）
    for i in range(5, 20):
        # 每个数值特征服从均值为0、标准差为1的正态分布
        data[f'feat_{i}'] = np.random.normal(loc=0, scale=1, size=n_samples)

    # 转换为Spark DataFrame（兼容pandas新版本）
    pdf = pd.DataFrame(data)
    # 使用to_dict('records')转换为字典列表，避免依赖iteritems
    spark_df = spark.createDataFrame(pdf.to_dict('records'))

    return spark_df

# 本地测试生成数据
if __name__ == "__main__":
    # 初始化SparkSession
    spark = SparkSession.builder \
        .appName("GenerateTestDataCSV") \
        .master("local[*]") \
        .getOrCreate()

    try:
        # 生成测试数据
        test_df = generate_test_data(spark, n_samples=5000)

        # 展示数据信息
        print("测试数据前5行：")
        test_df.show(5)
        print(f"测试数据总量: {test_df.count()} 行")

        # 定义保存路径
        save_path = "test_data.csv"
        print(f"准备保存数据到: {os.path.abspath(save_path)}")

        # 保存为CSV格式（header=True表示包含表头）
        test_df.write.mode('overwrite').option("header", "true").csv(save_path)

        # 验证文件是否存在（Spark会创建文件夹，检查文件夹是否存在）
        if os.path.exists(save_path):
            print(f"数据已成功保存至: {os.path.abspath(save_path)}")
            # 列出保存的文件
            print("保存的文件列表:")
            for file in os.listdir(save_path):
                print(f"- {file}")
        else:
            print(f"警告: 未找到保存的文件，但Spark未报告错误")

    except Exception as e:
        print(f"执行过程中发生错误: {str(e)}")
    finally:
        # 停止SparkSession
        spark.stop()
