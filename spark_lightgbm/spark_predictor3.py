from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.functions import col, when, lit, pandas_udf
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import os
import shutil

# 初始化 SparkSession
def init_spark():
    return SparkSession.builder \
        .appName("LightGBMGroupedPrediction") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

# 生成全量训练数据（含分类特征）
def generate_training_data(sample_size=100000):
    """生成大规模训练数据，包含数值特征和分类特征"""
    # 生成数值特征（15个）
    X, y = make_classification(
        n_samples=sample_size, n_features=15, n_informative=10,
        n_redundant=5, random_state=42
    )

    # 转换为DataFrame
    df = pd.DataFrame(X, columns=[f'num_feat_{i}' for i in range(15)])

    # 添加分类特征（5个，每个有3-5个类别）
    for i in range(5):
        n_categories = 3 + (i % 3)  # 3,4,5,3,4
        df[f'cat_feat_{i}'] = np.random.choice(
            [f'cat_{j}' for j in range(n_categories)],
            size=sample_size
        )

    # 添加标签和ID
    df['label'] = y
    df['id'] = [f'train_{i:08d}' for i in range(sample_size)]

    return df

# 训练LightGBM模型并保存
def train_lightgbm_model(model_path, sample_size=100000):
    """训练模型并返回分类特征信息"""
    # 生成训练数据
    df = generate_training_data(sample_size)

    # 区分分类特征和数值特征
    cat_features = [f'cat_feat_{i}' for i in range(5)]
    num_features = [f'num_feat_{i}' for i in range(15)]
    feature_names = cat_features + num_features

    # 对分类特征进行编码
    for feat in cat_features:
        df[feat] = df[feat].astype('category').cat.codes

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        df[feature_names], df['label'], test_size=0.2, random_state=42
    )

    # 构建LightGBM数据集
    lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train, categorical_feature=cat_features)

    # 训练参数
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': 1,
        'seed': 42
    }

    # 训练模型
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        num_boost_round=100
    )

    # 保存模型
    if os.path.exists(model_path):
        if os.path.isfile(model_path):
            os.remove(model_path)
        else:
            shutil.rmtree(model_path, ignore_errors=True)
    model.save_model(model_path)
    print(f"模型已保存至: {model_path}")

    # 返回训练集的分类特征信息
    cat_info = {}
    original_df = generate_training_data(sample_size=100)
    for feat in cat_features:
        original_cats = original_df[feat].unique().tolist()
        cat_info[feat] = {
            'original': original_cats,
            'encoded': list(range(len(original_cats)))
        }
    return cat_info

# 生成全量测试数据
def generate_test_data(spark, feature_names, cat_info, sample_size=20000):
    """生成与训练数据结构一致的测试数据"""
    np.random.seed(43)
    data = {
        "id": [f"test_{i:08d}" for i in range(sample_size)],
    }

    # 明确区分分类特征和数值特征
    cat_features = list(cat_info.keys())
    num_features = [f for f in feature_names if f not in cat_features]

    # 生成数值特征
    for feat in num_features:
        data[feat] = np.random.normal(loc=0.1, scale=1.2, size=sample_size)

    # 生成分类特征
    for feat in cat_features:
        valid_cats = cat_info[feat]['original']
        choices = []
        for _ in range(sample_size):
            if np.random.random() < 0.8:
                choices.append(np.random.choice(valid_cats))
            else:
                choices.append(f'unknown_{np.random.randint(1000)}')
        data[feat] = choices

    # 转换为pandas DataFrame
    pdf = pd.DataFrame(data)

    # 定义Spark Schema
    schema_fields = [StructField("id", StringType(), True)]
    for feat in num_features:
        schema_fields.append(StructField(feat, DoubleType(), True))
    for feat in cat_features:
        schema_fields.append(StructField(feat, StringType(), True))
    schema = StructType(schema_fields)

    return spark.createDataFrame(pdf, schema=schema)

# 特征预处理工具类
class FeatureProcessor:
    def __init__(self, cat_info):
        self.cat_info = cat_info

    def process(self, spark_df):
        """处理Spark DataFrame，将未知分类值替换为'unknown'"""
        processed_df = spark_df
        for feat in self.cat_info.keys():
            valid_cats = self.cat_info[feat]['original']
            processed_df = processed_df.withColumn(
                feat,
                when(col(feat).isin(valid_cats), col(feat)).otherwise(lit('unknown'))
            )
        return processed_df

# LightGBM预测器类
class LightGBMPredictor:
    def __init__(self, model_path, cat_info):
        self.model = lgb.Booster(model_file=model_path)
        self.feature_names = self.model.feature_name()
        self.cat_features = list(cat_info.keys())
        self.num_features = [f for f in self.feature_names if f not in self.cat_features]
        self.cat_info = cat_info
        # 定义输出schema
        self.output_schema = StructType([
            StructField("id", StringType(), True),
            StructField("prediction", IntegerType(), True),
            StructField("probability", DoubleType(), True)
        ])

    def predict(self, spark_df):
        """使用推荐的applyInPandas API进行预测"""
        # 广播模型和分类信息到所有节点
        model_broadcast = spark_df._sc.broadcast(self.model)
        cat_info_broadcast = spark_df._sc.broadcast(self.cat_info)
        feature_names = self.feature_names
        cat_features = self.cat_features
        num_features = self.num_features

        def predict_batch(pdf_iter):
            """批处理预测函数"""
            model = model_broadcast.value
            cat_info = cat_info_broadcast.value

            for pdf in pdf_iter:
                if len(pdf) == 0:
                    yield pd.DataFrame(columns=['id', 'prediction', 'probability'])
                    continue

                # 提取ID和特征
                ids = pdf[['id']].copy()
                X = pdf[feature_names].copy()

                # 处理分类特征
                for feat in cat_features:
                    cat_map = {cat: idx for idx, cat in enumerate(cat_info[feat]['original'])}
                    cat_map['unknown'] = -1
                    X[feat] = X[feat].map(cat_map).fillna(-1).astype(int)

                # 处理数值特征
                for feat in num_features:
                    if X[feat].dtype == 'object':
                        X[feat] = pd.to_numeric(X[feat], errors='coerce').fillna(0)

                # 执行预测并确保结果是原生Python类型
                probabilities = model.predict(X).tolist()  # 转换为Python列表
                predictions = [(1 if p > 0.5 else 0) for p in probabilities]  # 确保是Python int

                # 构造结果
                result = ids.copy()
                result['prediction'] = predictions
                result['probability'] = probabilities

                yield result

        # 使用推荐的applyInPandas替代GROUPED_MAP
        spark_df = spark_df.withColumn("group_key", col("id").substr(0, 2))
        result_df = spark_df.groupby("group_key").applyInPandas(
            predict_batch,
            schema=self.output_schema
        )

        return result_df.drop("group_key")

# 主函数
def main():
    model_path = "lightgbm_model.txt"
    spark = init_spark()

    try:
        # 1. 训练模型并获取分类特征信息
        print("开始训练模型...")
        cat_info = train_lightgbm_model(model_path, sample_size=100000)

        # 2. 生成测试数据
        print("生成测试数据...")
        predictor = LightGBMPredictor(model_path, cat_info)
        test_df = generate_test_data(
            spark,
            predictor.feature_names,
            cat_info,
            sample_size=20000
        )

        # 3. 预处理测试数据
        print("预处理测试数据...")
        processor = FeatureProcessor(cat_info)
        processed_test_df = processor.process(test_df)

        # 4. 执行预测
        print("执行预测...")
        result_df = predictor.predict(processed_test_df)

        # 5. 展示结果示例
        print("预测结果示例:")
        result_df.show(10)

        # 6. 保存结果
        result_df.write.mode("overwrite").csv("prediction_results", header=True)
        print("预测结果已保存至: prediction_results")

    except Exception as e:
        print(f"执行出错: {str(e)}")
    finally:
        spark.stop()
        print("Spark会话已关闭")

if __name__ == "__main__":
    main()
