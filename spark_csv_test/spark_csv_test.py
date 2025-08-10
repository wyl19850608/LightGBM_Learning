import os
from pyspark.sql import SparkSession
from pyspark.sql.types import (StructType, StructField, StringType,
                               IntegerType, DoubleType, DateType, BooleanType)
from pyspark.sql.functions import col, when




def create_test_csv(file_path):
    """创建包含各种测试数据的CSV文件"""
    with open(file_path, 'w') as f:
        f.write("id,name,age,salary,is_active,join_date,bonus\n")
        f.write("1,Alice,30,50000.50,true,2020-01-15,1000.0\n")
        f.write("2,Bob,,60000.75,false,2019-03-22,\n")  # 空年龄，空奖金
        f.write("3,Charlie,35,NaN,true,2021/05/30,2000\n")  # NaN薪资，非标准日期格式
        f.write("4,,40,75000.0,,2022-07-01,3000.50\n")  # 空姓名，空日期
        f.write("5,David,forty,80000.25,yes,2023-09-10,\n")  # 字符串类型的年龄，非标准布尔值
        f.write("6,Eve,50,,no,2018-11-11,5000\n")  # 空薪资

def main():
    # 显式创建SparkSession并赋值给spark变量
    spark = SparkSession.builder \
        .appName("CSVTypeHandlingTest") \
        .master("local[*]")  \
        .getOrCreate()

    # 创建测试CSV文件
    test_file = "test_data.csv"
    create_test_csv(test_file)

    # 定义手动Schema
    custom_schema = StructType([
        StructField("id", IntegerType(), nullable=True),
        StructField("name", StringType(), nullable=True),
        StructField("age", IntegerType(), nullable=True),
        StructField("salary", DoubleType(), nullable=True),
        StructField("is_active", BooleanType(), nullable=True),
        StructField("join_date", DateType(), nullable=True),
        StructField("bonus", DoubleType(), nullable=True)
    ])

    # 先按字符串读取所有字段
    string_schema = StructType([
        StructField(col_name, StringType(), nullable=True)
        for col_name in custom_schema.fieldNames()
    ])

    # 使用spark变量读取CSV
    df_string = spark.read \
        .format("csv") \
        .option("header", "true") \
        .option("inferSchema", "false") \
        .schema(string_schema) \
        .load(test_file)

    # 数据清洗和类型转换
    df_cleaned = df_string \
        .withColumn("id", col("id").cast(IntegerType())) \
        .withColumn("age", when(col("age").rlike("^\\d+$"), col("age").cast(IntegerType()))) \
        .withColumn("salary", when(col("salary") == "NaN", None)
                    .otherwise(col("salary").cast(DoubleType()))) \
        .withColumn("is_active", when(col("is_active").isin("true", "yes"), True)
                    .when(col("is_active").isin("false", "no"), False)
                    .otherwise(None)) \
        .withColumn("join_date",
                    when(col("join_date").rlike("\\d{4}-\\d{2}-\\d{2}"),
                         col("join_date").cast(DateType()))
                    .when(col("join_date").rlike("\\d{4}/\\d{2}/\\d{2}"),
                          col("join_date").cast(DateType()))) \
        .withColumn("bonus", col("bonus").cast(DoubleType()))

    print("=== 清洗后的DataFrame ===")
    df_cleaned.printSchema()
    df_cleaned.show()

    # 显示空值统计 - 修复了命名冲突问题
    print("\n=== 空值统计 ===")
    # 使用column_name作为循环变量，避免与col函数冲突
    null_counts = {column_name: df_cleaned.filter(col(column_name).isNull()).count()
                   for column_name in df_cleaned.columns}
    for col_name, count in null_counts.items():
        print(f"{col_name}: {count} 个空值")

    # 清理测试文件
    os.remove(test_file)

    # 停止SparkSession
    spark.stop()

if __name__ == "__main__":
    main()
