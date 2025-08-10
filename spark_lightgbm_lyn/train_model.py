import os
import time
import warnings
import joblib
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from pyspark.sql import SparkSession

from config import DATA_CONFIG, MODEL_CONFIG, SPARK_CONFIG, LIGHTGBM_PARAMS

warnings.filterwarnings("ignore")


def load_processed_data(spark, data_path):
    """加载处理后的训练数据（Spark DataFrame）"""
    print(f"加载处理后的数据（路径：{data_path}）...")
    df = spark.read.csv(
        data_path,
        header=True,
        inferSchema=True,
        quote='"',
        escape='"'
    )
    print(f"数据加载完成（行数：{df.count()}，列数：{len(df.columns)}）")
    return df


def prepare_training_data(spark_df, meta_data):
    """准备训练数据（转换为Pandas格式，拆分特征和标签）"""
    # 转换为Pandas DataFrame（小数据量场景）
    pdf = spark_df.toPandas()

    # 特征列和标签列
    label_col = meta_data["label_column"]
    feature_cols = meta_data["cat_features"] + meta_data["numeric_features"]

    # 过滤有效列（排除不存在的列）
    feature_cols = [col for col in feature_cols if col in pdf.columns]
    print(f"用于训练的特征数：{len(feature_cols)}（分类特征：{len(meta_data['cat_features'])}）")

    # 填充缺失值
    pdf[meta_data["numeric_features"]] = pdf[meta_data["numeric_features"]].fillna(0.0)
    pdf[meta_data["cat_features"]] = pdf[meta_data["cat_features"]].fillna("unknown")

    # 拆分训练集和验证集
    X = pdf[feature_cols]
    y = pdf[label_col].astype(int)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val, feature_cols


def train_lightgbm_model(X_train, X_val, y_train, y_val, cat_features, params):
    """训练LightGBM模型，兼容旧版本参数"""
    # 提前导入lightgbm模块，确保lgb变量在整个函数中可用
    import lightgbm as lgb
    print("\n开始训练LightGBM模型...")

    # 构造数据集（此时lgb已定义）
    lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train, categorical_feature=cat_features)

    # 准备训练参数，使用旧版本兼容的参数
    train_params = {
        'params': params,
        'train_set': lgb_train,
        'valid_sets': [lgb_train, lgb_val],
        'valid_names': ["train", "val"],
        'num_boost_round': params["num_boost_round"],
        'verbose_eval': 10  # 只使用旧版本支持的参数
    }

    # 检查LightGBM版本
    version = lgb.__version__
    print(f"LightGBM版本: {version}")

    # 训练模型
    model = lgb.train(** train_params)

    # 评估模型
    y_pred_proba = model.predict(X_val)
    y_pred = (y_pred_proba > 0.5).astype(int)
    print(f"\n验证集AUC：{roc_auc_score(y_val, y_pred_proba):.4f}")
    print("\n验证集分类报告：")
    print(classification_report(y_val, y_pred))

    return model




def save_model(model, model_path):
    """保存LightGBM模型"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    print(f"模型已保存至：{model_path}")


def main():
    try:
        start_time = time.time()
        print(">>> 开始模型训练流程 <<<")

        # 1. 初始化Spark
        spark = (SparkSession.builder
                 .appName(SPARK_CONFIG["appName"] + "_Train")
                 .master(SPARK_CONFIG["master"])
                 .config("spark.driver.memory", SPARK_CONFIG["driver_memory"])
                 .getOrCreate())

        # 2. 加载元数据
        print(f"加载元数据（路径：{MODEL_CONFIG['meta_data']}）...")
        meta_data = joblib.load(MODEL_CONFIG["meta_data"])
        cat_features = meta_data["cat_features"]

        # 3. 加载处理后的训练数据
        processed_df = load_processed_data(spark, DATA_CONFIG["processed_train_data"])

        # 4. 准备训练数据
        X_train, X_val, y_train, y_val, feature_cols = prepare_training_data(processed_df, meta_data)

        # 5. 训练模型
        model = train_lightgbm_model(
            X_train, X_val, y_train, y_val,
            cat_features=[f for f in cat_features if f in feature_cols],
            params=LIGHTGBM_PARAMS
        )

        # 6. 保存模型
        save_model(model, MODEL_CONFIG["model_path"])

        # 7. 输出训练信息
        elapsed = time.time() - start_time
        print(f"\n>>> 模型训练流程完成 <<<")
        print(f"总耗时：{elapsed:.2f}秒")

    except Exception as e:
        print(f"\n!!! 模型训练失败：{str(e)} !!!")
        raise
    finally:
        if 'spark' in locals():
            spark.stop()
            print("Spark会话已关闭")


if __name__ == "__main__":
    main()