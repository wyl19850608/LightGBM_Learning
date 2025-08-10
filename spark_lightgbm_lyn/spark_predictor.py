from pyspark.sql import SparkSession
from pyspark.sql.pandas.functions import PandasUDFType
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, BooleanType
from pyspark.sql.functions import pandas_udf, col, when, lit
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

# 生成示例训练数据（含分类特征）
def generate_training_data():
    # 生成10000样本，20特征（含5个分类特征）
    X, y = make_classification(
        n_samples=10000, n_features=15, n_informative=10,
        n_redundant=5, random_state=42
    )

    # 转换为DataFrame
    df = pd.DataFrame(X, columns=[f'num_feat_{i}' for i in range(15)])
    # 添加分类特征
    for i in range(5):
        df[f'cat_feat_{i}'] = np.random.choice([f'cat_{j}' for j in range(3)], size=10000)
    df['label'] = y
    df['id'] = [f'train_{i}' for i in range(10000)]

    return df

# 训练LightGBM模型并保存
def train_lightgbm_model(model_path):
    # 生成训练数据
    df = generate_training_data()
    # 区分分类特征和数值特征
    cat_features = [f'cat_feat_{i}' for i in range(5)]
    num_features = [f'num_feat_{i}' for i in range(15)]
    feature_names = cat_features + num_features

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        df[feature_names], df['label'], test_size=0.2, random_state=42
    )

    # 构建LightGBM数据集（指定分类特征）
    lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train, categorical_feature=cat_features)

    # 训练参数
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': 1
    }

    # 训练模型
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        num_boost_round=100
        # ,
        # early_stopping_rounds=10,
        # categorical_feature=cat_features
    )

    # 保存模型
    if os.path.exists(model_path):
        shutil.rmtree(model_path, ignore_errors=True)
    model.save_model(model_path)
    print(f"模型已保存至: {model_path}")

    # 返回训练集的分类特征信息（用于验证集对齐）
    cat_info = {feat: df[feat].unique().tolist() for feat in cat_features}
    return cat_info

# 生成测试数据（模拟可能存在特征不一致的情况）
def generate_test_data(spark, feature_names, cat_info):
    np.random.seed(43)  # 与训练集不同的随机种子，制造特征差异
    data = {
        "id": [f"test_{i}" for i in range(1000)],
    }

    # 生成数值特征
    for feat in feature_names:
        if feat.startswith('num_'):
            data[feat] = np.random.randn(1000)

    # 生成分类特征（故意加入训练集没有的类别）
    for feat in cat_info.keys():
        # 80%概率使用训练集存在的类别，20%概率使用新类别
        if np.random.random() < 0.8:
            data[feat] = np.random.choice(cat_info[feat], size=1000)
        else:
            data[feat] = np.random.choice([f'unknown_{i}' for i in range(2)], size=1000)

    # 转换为DataFrame
    pdf = pd.DataFrame(data)

    # 定义schema
    schema = StructType([
                            StructField("id", StringType(), True)
                        ] + [StructField(feat, StringType(), True) if feat.startswith('cat_')
                             else StructField(feat, DoubleType(), True) for feat in feature_names])

    return spark.createDataFrame(pdf, schema=schema)

# 特征预处理工具类（确保分类特征一致性）
class FeatureProcessor:
    def __init__(self, cat_info):
        self.cat_info = cat_info  # 训练集的分类特征信息

    def process(self, spark_df):
        """处理Spark DataFrame，确保分类特征与训练集一致"""
        processed_df = spark_df
        for feat, valid_cats in self.cat_info.items():
            # 将不在训练集类别的值替换为'unknown'
            processed_df = processed_df.withColumn(
                feat,
                when(col(feat).isin(valid_cats), col(feat)).otherwise(lit('unknown'))
            )
        return processed_df

# LightGBM预测器类
class LightGBMPredictor:
    def __init__(self, model_path):
        self.model = lgb.Booster(model_file=model_path)
        self.feature_names = self.model.feature_name()
        # 识别分类特征（假设特征名以'cat_'开头）
        self.cat_features = [f for f in self.feature_names if f.startswith('cat_')]

    def _get_output_schema(self):
        return StructType([
            StructField("id", StringType(), True),
            StructField("prediction", IntegerType(), True),
            StructField("probability", DoubleType(), True)
        ])

    def predict(self, spark_df):
        model = self.model
        feature_names = self.feature_names
        cat_features = self.cat_features

        @pandas_udf(self._get_output_schema(), PandasUDFType.GROUPED_MAP)
        def grouped_predict(pdf):
            try:
                if len(pdf) == 0:
                    return pd.DataFrame(columns=['id', 'prediction', 'probability'])

                # 提取ID和特征
                ids = pdf[['id']].copy()
                X = pdf[feature_names].copy()

                # 处理分类特征（转换为类别型）
                for feat in cat_features:
                    X[feat] = X[feat].astype('category')

                # 数值特征处理
                for feat in [f for f in feature_names if f not in cat_features]:
                    if X[feat].dtype == 'object':
                        X[feat] = pd.to_numeric(X[feat], errors='coerce').fillna(0)

                # 预测
                probabilities = model.predict(X)
                predictions = (probabilities > 0.5).astype(int)

                # 构造结果
                result = ids.copy()
                result['prediction'] = predictions
                result['probability'] = probabilities
                return result
            except Exception as e:
                print(f"分组预测出错: {str(e)}")
                return pd.DataFrame(columns=['id', 'prediction', 'probability'])

        # 分组策略：按id前2位分组
        spark_df = spark_df.withColumn("group_key", col("id").substr(0, 2))
        result_df = spark_df.groupby("group_key").apply(grouped_predict)
        return result_df.drop("group_key")

# 主函数
def main():
    model_path = "lightgbm_model.txt"
    spark = init_spark()

    try:
        # 1. 训练模型并获取分类特征信息
        print("开始训练模型...")
        cat_info = train_lightgbm_model(model_path)

        # 2. 生成测试数据
        print("生成测试数据...")
        predictor = LightGBMPredictor(model_path)
        test_df = generate_test_data(spark, predictor.feature_names, cat_info)

        # 3. 预处理测试数据（确保分类特征一致）
        print("预处理测试数据...")
        processor = FeatureProcessor(cat_info)
        processed_test_df = processor.process(test_df)

        # 4. 执行预测
        print("执行预测...")
        result_df = predictor.predict(processed_test_df)

        # 5. 展示结果
        print("预测结果示例:")
        result_df.show(10)

        # 6. 保存结果（可选）
        result_df.write.mode("overwrite").csv("prediction_results", header=True)
        print("预测结果已保存至: prediction_results")

    except Exception as e:
        print(f"执行出错: {str(e)}")
    finally:
        spark.stop()
        print("Spark会话已关闭")

if __name__ == "__main__":
    main()