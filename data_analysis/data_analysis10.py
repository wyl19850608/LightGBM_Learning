import os
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.stat import Correlation, ChiSquareTest
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType, BooleanType

# 设置PySpark环境变量
os.environ["PYSPARK_PYTHON"] = "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10"

# 创建SparkSession
spark = SparkSession.builder \
    .appName("DataAnalysis") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .getOrCreate()

# 设置日志级别以减少警告信息
spark.sparkContext.setLogLevel("ERROR")

print("=== 数据生成与准备 ===")
# 定义可能的字符串类别
categories = ["Aa", "Bb", "Cc", "Dd", "Ee"]

# 完全使用Python原生随机数生成，避免NumPy
import random

null_probabilities = {
    "id": 0.0,          # ID列不设置空值
    "feature1": 0.30,   # 30%概率为空
    "feature2": 0.03,   # 3%概率为空
    "category": 0.02,   # 2%概率为空
    "string_col1": 0.10, # 10%概率为空
    "string_col2": 0.10, # 10%概率为空
    "boolean_col1": 0.08, # 8%概率为空
    "boolean_col2": 0.08  # 8%概率为空
}

# 生成示例数据，随机添加空值
data = []
for i in range(1000):
    # 生成各列值，根据概率决定是否设为空
    id_val = i
    feature1_val = random.normalvariate(0, 1) if random.random() > null_probabilities["feature1"] else None
    feature2_val = random.normalvariate(0, 1) if random.random() > null_probabilities["feature2"] else None
    category_val = random.randint(1, 5) if random.random() > null_probabilities["category"] else None
    string_val1 = random.choice(categories) if random.random() > null_probabilities["string_col1"] else None
    string_val2 = random.choice(categories) if random.random() > null_probabilities["string_col2"] else None
    boolean_val1 = random.choice([True, False]) if random.random() > null_probabilities["boolean_col1"] else None
    boolean_val2 = random.choice([True, False]) if random.random() > null_probabilities["boolean_col2"] else None

    data.append((id_val, feature1_val, feature2_val, category_val, string_val1, string_val2, boolean_val1, boolean_val2))

# 定义Schema，明确各列类型
schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("feature1", DoubleType(), True),
    StructField("feature2", DoubleType(), True),
    StructField("category", IntegerType(), True),
    StructField("string_col1", StringType(), True),
    StructField("string_col2", StringType(), True),
    StructField("boolean_col1", BooleanType(), True),
    StructField("boolean_col2", BooleanType(), True)
])

# 创建DataFrame
df = spark.createDataFrame(data, schema)

# 添加随机label列(0/1)，使用Spark原生函数
df = df.withColumn("label", (F.rand() > 0.5).cast("integer"))

print("\n=== 数据基本信息 ===")
print(f"数据行数: {df.count()}")
print(f"数据列数: {len(df.columns)}")
df.printSchema()
df.show(5, truncate=False)

print("\n=== 缺失值分析 ===")
missing_counts = []
for c, t in df.dtypes:
    if t in ['double', 'float', 'int']:
        count_expr = F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c)
    else:
        count_expr = F.count(F.when(F.col(c).isNull(), c)).alias(c)
    count_result = df.select(count_expr).collect()[0][0]
    missing_counts.append(count_result)

missing_counts_df = spark.createDataFrame([missing_counts], df.columns)
missing_counts_df.show()

missing_pct = []
for c in df.columns:
    pct_expr = (F.col(c) / df.count() * 100).alias(f"{c}_missing_pct")
    pct_result = missing_counts_df.select(pct_expr).collect()[0][0]
    missing_pct.append(pct_result)

missing_pct_df = spark.createDataFrame([missing_pct], [f"{c}_missing_pct" for c in df.columns])
missing_pct_df.show()

print("\n=== 基本统计量分析 ===")
numeric_cols = [c for c, t in df.dtypes if t in ['int', 'double'] and c != 'label']
for col_name in numeric_cols:
    summary = df.select(col_name).summary("count", "mean", "stddev", "min", "max")
    print(f"列 {col_name} 的统计量:")
    summary.show()

print("\n=== 变量与label的相关性和卡方检验 ===")
# 连续变量与label的皮尔逊相关性
print("\n--- 连续变量与label的皮尔逊相关性 ---")
for col_name in numeric_cols:
    corr_value = df.stat.corr(col_name, "label")
    print(f"{col_name} 与 label 的皮尔逊相关系数: {corr_value:.4f}")

# 获取所有分类列（字符串列、布尔列和category列）
string_cols = [c for c, t in df.dtypes if t == 'string']
bool_cols = [c for c, t in df.dtypes if t == 'boolean']
cat_cols = string_cols + bool_cols + ['category']

# 存储分类特征的重要性得分（使用负对数p值）
cat_importance = {}

print("\n--- 分类变量与label的卡方检验和相关性 ---")
for col_name in cat_cols:
    if col_name in bool_cols:
        # 处理布尔列
        encoded_col = f"{col_name}_int"
        df = df.withColumn(encoded_col, F.col(col_name).cast("integer"))

        # 创建干净的DataFrame用于分析
        clean_df = df.select(encoded_col, "label").dropna()

        # 计算相关性
        corr_value = clean_df.stat.corr(encoded_col, "label")
        print(f"{col_name} 与 label 的相关性: {corr_value:.4f}")

        # 执行卡方检验
        assembler = VectorAssembler(
            inputCols=[encoded_col],
            outputCol=f"{col_name}_vector",
            handleInvalid="skip"
        )
        vector_df = assembler.transform(clean_df)

        chi_result = ChiSquareTest.test(vector_df, f"{col_name}_vector", "label").head()
        p_value = chi_result.pValues[0] if hasattr(chi_result, "pValues") else chi_result.pValue
        print(f"{col_name} 与 label 的卡方检验 p值: {p_value:.4f}")

        # 计算重要性得分
        cat_importance[col_name] = -1 * p_value if p_value > 0 else float('inf')

    elif col_name == 'category':
        # 处理category列（已经是整数）
        clean_df = df.select(col_name, "label").dropna()

        # 计算相关性
        corr_value = clean_df.stat.corr(col_name, "label")
        print(f"{col_name} 与 label 的相关性: {corr_value:.4f}")

        # 执行卡方检验
        assembler = VectorAssembler(
            inputCols=[col_name],
            outputCol=f"{col_name}_vector",
            handleInvalid="skip"
        )
        vector_df = assembler.transform(clean_df)

        chi_result = ChiSquareTest.test(vector_df, f"{col_name}_vector", "label").head()
        p_value = chi_result.pValues[0] if hasattr(chi_result, "pValues") else chi_result.pValue
        print(f"{col_name} 与 label 的卡方检验 p值: {p_value:.4f}")

        # 计算重要性得分
        cat_importance[col_name] = -1 * p_value if p_value > 0 else float('inf')

    else:
        # 处理字符串列
        indexed_col = f"{col_name}_indexed"

        # 检查索引列是否已存在
        if indexed_col not in df.columns:
            indexer = StringIndexer(inputCol=col_name, outputCol=indexed_col, handleInvalid='skip')
            df = indexer.fit(df).transform(df)

        # 创建干净的DataFrame用于分析
        clean_df = df.select(indexed_col, "label").dropna()

        # 计算相关性
        corr_value = clean_df.stat.corr(indexed_col, "label")
        print(f"{col_name} 与 label 的相关性: {corr_value:.4f}")

        # 执行卡方检验
        assembler = VectorAssembler(
            inputCols=[indexed_col],
            outputCol=f"{col_name}_vector",
            handleInvalid="skip"
        )
        vector_df = assembler.transform(clean_df)

        chi_result = ChiSquareTest.test(vector_df, f"{col_name}_vector", "label").head()
        p_value = chi_result.pValues[0] if hasattr(chi_result, "pValues") else chi_result.pValue
        print(f"{col_name} 与 label 的卡方检验 p值: {p_value:.4f}")

        # 计算重要性得分
        cat_importance[col_name] = -1 * p_value if p_value > 0 else float('inf')

print("\n--- 变量间相关性矩阵可视化 ---")
corr_matrix = []
plot_numeric_cols = numeric_cols + ['label']  # 包含label列用于可视化

for i in range(len(plot_numeric_cols)):
    row = []
    for j in range(len(plot_numeric_cols)):
        if i == j:
            row.append(1.0)
        else:
            corr_value = df.stat.corr(plot_numeric_cols[i], plot_numeric_cols[j])
            row.append(corr_value)
    corr_matrix.append(row)

# 使用Pandas和Matplotlib可视化
plt.figure(figsize=(10, 8))
corr_df = pd.DataFrame(corr_matrix, columns=plot_numeric_cols, index=plot_numeric_cols)
plt.imshow(corr_df.values, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=45)
plt.yticks(range(len(corr_df.columns)), corr_df.columns)
plt.title("Correlation Matrix (Pearson)")
plt.tight_layout()
plt.savefig("correlation_matrix_pearson.png")
print("皮尔逊相关性矩阵已保存为: correlation_matrix_pearson.png")

print("\n=== 类别型特征的分布分析 ===")
# 类别型特征分布
for col_name in cat_cols:
    print(f"\n{col_name} 的分布:")
    df.groupBy(col_name).count().show()

print("\n=== 特征与标签的交叉表分析 ===")
# 创建交叉表
for col_name in string_cols + bool_cols:
    print(f"\n{col_name} 与 label 的交叉表:")
    df.crosstab(col_name, "label").show()

print("\n=== 特征对label的重要性分析 ===")
# 准备特征工程的步骤
stages = []

# 对字符串列进行索引编码
for col_name in string_cols:
    indexed_col = f"{col_name}_indexed"
    if indexed_col not in df.columns:
        indexer = StringIndexer(inputCol=col_name, outputCol=indexed_col, handleInvalid='skip')
        stages.append(indexer)

# 对布尔列进行转换
for col_name in bool_cols:
    int_col = f"{col_name}_int"
    if int_col not in df.columns:
        df = df.withColumn(int_col, F.col(col_name).cast("integer"))

# 定义所有特征列，明确排除label列
feature_cols = numeric_cols + \
               [f"{col}_indexed" for col in string_cols] + \
               [f"{col}_int" for col in bool_cols]

# 创建向量组装器
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
stages.append(assembler)

# 创建处理缺失值的DataFrame
df_clean = df.na.drop(subset=feature_cols + ["label"])

# 创建并运行预处理pipeline
pipeline = Pipeline(stages=stages)
pipeline_model = pipeline.fit(df_clean)
df_transformed = pipeline_model.transform(df_clean)

# 训练随机森林模型以获取特征重要性
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100, seed=42)
rf_model = rf.fit(df_transformed)

# 获取特征重要性
feature_importance = rf_model.featureImportances.toArray()

# 创建特征重要性DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': feature_importance
})

# 添加分类特征的重要性（基于卡方检验）
for col, importance in cat_importance.items():
    if col in feature_cols:
        importance_df.loc[importance_df['Feature'] == col, 'Importance'] = importance

# 映射特征名到原始列名
def map_feature_to_original(feature_name):
    if feature_name.endswith("_indexed"):
        return feature_name.replace("_indexed", "")
    elif feature_name.endswith("_int"):
        return feature_name.replace("_int", "")
    else:
        return feature_name

importance_df['Original_Feature'] = importance_df['Feature'].apply(map_feature_to_original)

# 按原始特征名分组并计算平均重要性
grouped_importance = importance_df.groupby('Original_Feature')['Importance'].mean().reset_index()

# 按重要性排序
sorted_importance = grouped_importance.sort_values('Importance', ascending=False)

# 输出结果
print("\n特征对label的重要性排序（使用原始列名）:")
print(sorted_importance)

# 可视化特征重要性
plt.figure(figsize=(12, 8))
plt.barh(sorted_importance['Original_Feature'], sorted_importance['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance for Label')
plt.tight_layout()
plt.savefig("feature_importance.png")
print("特征重要性图已保存为: feature_importance.png")

# 输出全量数据到CSV文件
print("\n=== 输出全量数据 ===")
output_path = "full_data.csv"
df.toPandas().to_csv(output_path, index=False)
print(f"全量数据已保存到: {output_path}")

# 停止SparkSession
spark.stop()
print("\n=== 分析完成 ===")