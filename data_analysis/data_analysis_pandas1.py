import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.stat import Correlation, ChiSquareTest
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType, BooleanType
from pyspark import SparkConf, SparkContext
import random

# 设置PySpark环境变量
os.environ['PYSPARK_SUBMIT_ARGS'] = '--conf "spark.driver.extraJavaOptions=-XX:+IgnoreUnrecognizedVMOptions --add-opens=java.base/sun.nio.ch=ALL-UNNAMED" --conf "spark.executor.extraJavaOptions=-XX:+IgnoreUnrecognizedVMOptions --add-opens=java.base/sun.nio.ch=ALL-UNNAMED" pyspark-shell'
os.environ["PYSPARK_PYTHON"] = "/Users/wangyanling32/code/pythonProject4/.venv/bin/python3.8"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/Users/wangyanling32/code/pythonProject4/.venv/bin/python3.8"

# 创建SparkSession
spark = SparkSession.builder \
    .appName("DataAnalysis") \
    .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC --add-opens java.base/java.nio=ALL-UNNAMED") \
    .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC --add-opens java.base/java.nio=ALL-UNNAMED") \
    .getOrCreate()

# 设置日志级别
spark.sparkContext.setLogLevel("ERROR")

print("=== 数据生成与准备 ===")
categories = ["Aa", "Bb", "Cc", "Dd", "Ee"]

null_probabilities = {
    "id": 0.0,
    "feature1": 0.0,
    "feature2": 0.0,
    "category": 0.0,
    "boolean_col1": 0.0,
    "boolean_col2": 0.0
}

# 生成示例数据
data = []
for i in range(1000):
    id_val = i
    feature1_val = random.normalvariate(0, 1) if random.random() > null_probabilities["feature1"] else None
    feature2_val = random.normalvariate(0, 1) if random.random() > null_probabilities["feature2"] else None
    category_val = random.randint(1, 5) if random.random() > null_probabilities["category"] else None
    boolean_val1 = random.choice([True, False]) if random.random() > null_probabilities["boolean_col1"] else None
    boolean_val2 = random.choice([True, False]) if random.random() > null_probabilities["boolean_col2"] else None

    data.append((id_val, feature1_val, feature2_val, category_val, boolean_val1, boolean_val2))

# 定义Schema
schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("feature1", DoubleType(), True),
    StructField("feature2", DoubleType(), True),
    StructField("category", IntegerType(), True),
    StructField("boolean_col1", BooleanType(), True),
    StructField("boolean_col2", BooleanType(), True)
])

# 创建DataFrame
df = spark.createDataFrame(data, schema)

# 添加label列
df = df.withColumn("label", (F.rand() > 0.5).cast("integer"))

print("\n=== 数据基本信息 ===")
print(f"数据行数: {df.count()}")
print(f"数据列数: {len(df.columns)}")
df.printSchema()
df.show(5, truncate=False)

# 解决布尔类型转换问题 - 方法1：转换为整数类型
boolean_columns = [col for col, dtype in df.dtypes if dtype == 'boolean']
for col in boolean_columns:
    df = df.withColumn(col, F.col(col).cast("integer"))

# 解决布尔类型转换问题 - 方法2：临时修改PySpark的类型映射函数
# 注意：如果使用方法1，此代码可以注释掉
# import pyspark.sql.pandas.conversion as pconv
# pconv._to_corrected_pandas_type = lambda dtype: bool if dtype == 'boolean' else dtype

# 将PySpark DataFrame转换为Pandas DataFrame
print("\n=== 将DataFrame转换为Pandas进行计算 ===")
pandas_df = df.toPandas()

# 1. 缺失值分析
print("\n--- 缺失值分析 ---")
missing_values = pandas_df.isnull().sum()
missing_percentage = (pandas_df.isnull().sum() / len(pandas_df)) * 100
missing_analysis = pd.concat([missing_values, missing_percentage], axis=1)
missing_analysis.columns = ['缺失值数量', '缺失值百分比(%)']
print(missing_analysis)

# 2. 基本统计量分析
print("\n--- 基本统计量分析 ---")
numeric_columns = pandas_df.select_dtypes(include=['number']).columns
statistics = pandas_df[numeric_columns].describe()
print(statistics)

# 3. 相关性分析
print("\n--- 相关性分析 ---")
correlation = pandas_df[numeric_columns].corr()
print(correlation)

# 4. 分类变量分布分析
print("\n--- 分类变量分布分析 ---")
# 由于布尔列已转换为整数，我们需要重新识别分类变量
categorical_columns = ['category']  # 手动指定分类列
for col in categorical_columns:
    print(f"\n{col} 的分布:")
    print(pandas_df[col].value_counts(dropna=False))

# 5. 特征与标签的交叉表分析
print("\n--- 特征与标签的交叉表分析 ---")
for col in categorical_columns:
    if col != 'label':  # 确保不与label列交叉
        cross_table = pd.crosstab(pandas_df[col], pandas_df['label'])
        print(f"\n{col} 与 label 的交叉表:")
        print(cross_table)

# 6. 数据可视化 - 连续变量分布
print("\n--- 连续变量分布可视化 ---")
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_columns):
    if col != 'label':  # 不绘制label列
        plt.subplot(2, 3, i+1)
        sns.histplot(pandas_df[col], kde=True)
        plt.title(f'{col} 分布')
plt.tight_layout()
plt.savefig("continuous_variable_distributions.png")
print("连续变量分布图已保存为: continuous_variable_distributions.png")

# 7. 数据可视化 - 特征与标签关系
print("\n--- 特征与标签关系可视化 ---")
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_columns):
    if col != 'label':  # 不绘制label列
        plt.subplot(2, 3, i+1)
        sns.scatterplot(x=col, y='label', data=pandas_df)
        plt.title(f'{col} 与 label 的关系')
plt.tight_layout()
plt.savefig("feature_label_relationships.png")
print("特征与标签关系图已保存为: feature_label_relationships.png")

# 8. 使用Pandas进行更复杂的分析 - 例如分组统计
print("\n--- 分组统计分析 ---")
grouped_data = pandas_df.groupby('label')[numeric_columns].mean()
print("按label分组的特征均值:")
print(grouped_data)

# 9. 使用Pandas进行特征重要性分析（基于随机森林）
print("\n--- 使用Pandas进行特征重要性分析 ---")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 准备特征和标签
X = pandas_df.drop('label', axis=1)
# 处理缺失值
X = X.fillna(X.mean())
# 对分类变量进行编码
X = pd.get_dummies(X, drop_first=True)
y = pandas_df['label']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练随机森林模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 获取特征重要性
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
})

# 按重要性排序
sorted_importance = feature_importance.sort_values('Importance', ascending=False)
print("\n特征重要性排序:")
print(sorted_importance.head(10))  # 只显示前10个重要特征

# 可视化特征重要性
plt.figure(figsize=(12, 8))
plt.barh(sorted_importance.head(10)['Feature'], sorted_importance.head(10)['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 10 Feature Importance for Label')
plt.tight_layout()
plt.savefig("pandas_feature_importance.png")
print("Pandas特征重要性图已保存为: pandas_feature_importance.png")

# 停止SparkSession
spark.stop()
print("\n=== 分析完成 ===")