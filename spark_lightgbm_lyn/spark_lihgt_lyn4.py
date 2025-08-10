import os
import time
import warnings
import joblib
import lightgbm as lgb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.functions import col, lit, when, size, expr, array_min, array_max
from pyspark.sql.utils import AnalysisException


# ------------------------------
# 配置与常量定义
# ------------------------------
warnings.filterwarnings("ignore")

# Spark配置（根据实际环境调整，此处为本地测试配置）
SPARK_CONFIG = {
    "appName": "LightGBMGroupedPrediction",
    "master": "local[*]",
    "driver_memory": "4g",
    "executor_memory": "4g",
    "arrow_enabled": "false"  # 禁用Arrow以兼容部分环境
}

# 特征与模型相关常量
REQUIRED_COLUMNS = ["pboc_repay_date_days"]  # 特征处理依赖的关键列
DEFAULT_NUMERIC_FILL = 0.0  # 数值特征缺失填充值
UNKNOWN_CATEGORY = "unknown"  # 未知分类特征填充值


# ------------------------------
# 工具类定义
# ------------------------------
class FeatureProcessor:
    """特征处理器：负责特征清洗、转换和衍生"""

    def __init__(self, cat_info):
        """
        :param cat_info: 分类特征元信息（包含每个特征的有效类别）
        """
        self.cat_info = cat_info

    def process(self, spark_df):
        """主处理流程：衍生特征 + 分类特征清洗"""
        # 1. 检查必要列是否存在
        self._check_required_columns(spark_df)

        # 2. 从日期列衍生统计特征（min/mean/max）
        processed_df = self._derive_date_stats(spark_df)

        # 3. 清洗分类特征（替换无效类别为unknown）
        processed_df = self._clean_categorical_features(processed_df)

        return processed_df

    def _check_required_columns(self, spark_df):
        """检查必要列是否存在，不存在则抛出异常"""
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in spark_df.columns]
        if missing_cols:
            raise ValueError(f"输入数据缺少必要列: {missing_cols}")

    def _derive_date_stats(self, spark_df):
        """从pboc_repay_date_days衍生统计特征"""
        # 分割字符串为数组并计算统计量
        date_stats_df = spark_df.withColumn(
            "split_array",
            expr("filter(transform(split(pboc_repay_date_days, ',,'), x -> cast(trim(x) as int)), y -> y is not null)")
        ).withColumn(
            "min_repay_day",
            when(col("split_array").isNotNull() & (size(col("split_array")) > 0),
                 array_min(col("split_array"))).otherwise(None)
        ).withColumn(
            "max_repay_day",
            when(col("split_array").isNotNull() & (size(col("split_array")) > 0),
                 array_max(col("split_array"))).otherwise(None)
        ).withColumn(
            "mean_repay_day",
            when(col("split_array").isNotNull() & (size(col("split_array")) > 0),
                 expr("round(aggregate(split_array, cast(0 as double), "
                      "(acc, x) -> acc + x, acc -> acc / size(split_array)))")).otherwise(None)
        ).select("pboc_repay_date_days", "min_repay_day", "mean_repay_day", "max_repay_day")

        # 关联统计特征并删除原始列
        result_df = spark_df.join(date_stats_df, on="pboc_repay_date_days", how="left")
        return result_df.drop("pboc_repay_date_days")

    def _clean_categorical_features(self, spark_df):
        """清洗分类特征：将不在有效类别中的值替换为unknown"""
        # 只处理数据中存在的分类特征
        existing_cat_features = [f for f in self.cat_info.keys() if f in spark_df.columns]

        for feature in existing_cat_features:
            valid_categories = self.cat_info[feature]["original"]
            spark_df = spark_df.withColumn(
                feature,
                when(col(feature).isin(valid_categories), col(feature)).otherwise(lit(UNKNOWN_CATEGORY))
            )
        return spark_df


class LightGBMPredictor:
    """LightGBM预测器：负责加载模型、批量预测"""

    def __init__(self, model_path, cat_features, cat_info):
        """
        :param model_path: LightGBM模型路径
        :param cat_features: 分类特征列表
        :param cat_info: 分类特征元信息
        """
        self.model = self._load_model(model_path)
        self.feature_names = self.model.feature_name()
        self.cat_features = [f for f in cat_features if f in self.feature_names]  # 过滤模型中存在的分类特征
        self.num_features = [f for f in self.feature_names if f not in self.cat_features]
        self.cat_info = cat_info
        self.output_schema = self._define_output_schema()

    def _load_model(self, model_path):
        """加载LightGBM模型，添加异常处理"""
        try:
            return lgb.Booster(model_file=model_path)
        except Exception as e:
            raise RuntimeError(f"加载模型失败（路径：{model_path}）：{str(e)}")

    def _define_output_schema(self):
        """定义预测结果的Schema"""
        return StructType([
            StructField("user_id", StringType(), True),
            StructField("unique_id", StringType(), True),
            StructField("prediction", DoubleType(), True)
        ])

    def predict(self, spark_df):
        """对Spark DataFrame进行批量预测"""
        # 1. 检查输入特征是否完整
        self._check_features_complete(spark_df)

        # 2. 广播模型和元信息（减少Executor间数据传输）
        model_broadcast = spark_df._sc.broadcast(self.model)
        cat_info_broadcast = spark_df._sc.broadcast(self.cat_info)
        cat_features_broadcast = spark_df._sc.broadcast(self.cat_features)
        num_features_broadcast = spark_df._sc.broadcast(self.num_features)
        feature_names = self.feature_names

        # 3. 定义批量预测函数（在Executor中执行）
        def _predict_batch(pdf):
            """处理单批Pandas DataFrame并返回预测结果"""
            if pdf.empty:
                return pd.DataFrame(columns=["user_id", "unique_id", "prediction"])

            # 提取广播变量
            model = model_broadcast.value
            cat_info = cat_info_broadcast.value
            cat_features = cat_features_broadcast.value
            num_features = num_features_broadcast.value

            # 准备特征数据
            feature_data = {}

            # 处理分类特征（映射为编码值）
            for feature in cat_features:
                if feature not in pdf.columns:
                    feature_data[feature] = [-1] * len(pdf)  # 缺失特征填充-1
                    continue
                # 类别映射：原始类别 -> 编码（unknown映射为-1）
                cat_map = {cat: idx for idx, cat in enumerate(cat_info[feature]["original"])}
                cat_map[UNKNOWN_CATEGORY] = -1
                feature_data[feature] = [cat_map.get(str(val), -1) for val in pdf[feature].fillna(UNKNOWN_CATEGORY)]

            # 处理数值特征（填充缺失值）
            for feature in num_features:
                if feature not in pdf.columns:
                    feature_data[feature] = [DEFAULT_NUMERIC_FILL] * len(pdf)
                    continue
                # 缺失值填充为默认值
                feature_data[feature] = pdf[feature].fillna(DEFAULT_NUMERIC_FILL).astype(float).tolist()

            # 构建特征DataFrame
            X = pd.DataFrame(feature_data, columns=feature_names)

            # 执行预测
            probabilities = model.predict(X)
            predictions = (probabilities > 0.5).astype(int)  # 二分类阈值判断

            # 构造结果DataFrame
            return pd.DataFrame({
                "user_id": pdf["user_id"].astype(str),
                "unique_id": pdf["unique_id"].astype(str),
                "prediction": predictions
            })

        # 4. 按group_key分组预测（避免单批数据过大）
        grouped_df = spark_df.withColumn("group_key", col("user_id").substr(0, 2))  # 按user_id前2位分组
        result_df = grouped_df.groupby("group_key").applyInPandas(
            _predict_batch,
            schema=self.output_schema
        )

        return result_df.drop("group_key")

    def _check_features_complete(self, spark_df):
        """检查输入数据是否包含所有必要特征"""
        missing_features = [f for f in self.feature_names if f not in spark_df.columns]
        if missing_features:
            raise ValueError(f"输入数据缺少模型必要特征：{missing_features}")


# ------------------------------
# 工具函数
# ------------------------------
def load_metadata(meta_path):
    """加载特征元数据（分类特征列表和类别信息）"""
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"元数据文件不存在：{meta_path}")
    try:
        meta_data = joblib.load(meta_path)
        return meta_data.get("cat_features", []), meta_data.get("cat_info", {})
    except Exception as e:
        raise RuntimeError(f"解析元数据失败（路径：{meta_path}）：{str(e)}")


# ------------------------------
# 主函数
# ------------------------------
def main():
    # 配置参数（可根据实际情况调整）
    config = {
        "model_path": "/home/finance/IDE/work/user_data/potential_custom_mining/feature_funcs/model/model_test_cp.txt",
        "meta_path": "cat_meta_data.pkl",
        "input_path": "data.csv",  # 输入数据路径
        "output_path": "prediction_results",  # 输出结果路径
        "sample_size": 5  # 打印样例数量
    }

    try:
        start_time = time.time()
        print(">>> 开始预测流程 <<<")

        # 1. 初始化Spark会话
        spark = (SparkSession.builder \
                 .appName("LightGBMGroupedPrediction") \
                 .master("local[*]") \
                 .config("spark.driver.memory", "4g") \
                 .config("spark.executor.memory", "4g") \
                 .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
                 .getOrCreate())
        print(f"Spark初始化完成（版本：{spark.version}）")

        # 2. 加载元数据和模型
        print(f"加载元数据（路径：{config['meta_path']}）...")
        cat_features, cat_info = load_metadata(config["meta_path"])

        print(f"加载模型（路径：{config['model_path']}）...")
        predictor = LightGBMPredictor(
            model_path=config["model_path"],
            cat_features=cat_features,
            cat_info=cat_info
        )
        print(f"模型加载完成（特征数：{len(predictor.feature_names)}，分类特征数：{len(predictor.cat_features)}）")

        # 3. 读取并预处理数据
        print(f"读取输入数据（路径：{config['input_path']}）...")
        raw_df = spark.read.csv(
            config["input_path"],
            header=True,
            inferSchema=True,
            quote='"',  # 处理带引号的字段
            escape='"'
        )

        # 修正列名（避免空列名，根据实际数据调整）
        # 注意：此处需确保columns_df与raw_df的列数一致，建议从元数据加载列名
        columns_df = raw_df.columns  # 使用原始列名（替换空列名逻辑）
        df = raw_df.toDF(*columns_df)

        # 4. 特征处理
        print("开始特征处理...")
        feature_processor = FeatureProcessor(cat_info=cat_info)
        processed_df = feature_processor.process(df)
        print(f"特征处理完成（处理后列数：{len(processed_df.columns)}）")

        # 5. 执行预测
        print("开始预测...")
        result_df = predictor.predict(processed_df)
        print(f"预测完成（结果行数：{result_df.count()}）")

        # 6. 保存并展示结果
        print(f"保存预测结果至：{config['output_path']}")
        result_df.write.mode("overwrite").csv(
            config["output_path"],
            header=True,
            quote='"',
            escape='"'
        )

        print(f"\n预测样例（前{config['sample_size']}行）：")
        result_df.show(config["sample_size"], truncate=False)

        # 7. 打印性能统计
        elapsed = time.time() - start_time
        print(f"\n>>> 预测流程完成 <<<")
        print(f"总耗时：{elapsed:.2f}秒")
        print(f"处理速度：{df.count() / elapsed:.2f}行/秒")

    except Exception as e:
        print(f"\n!!! 预测流程失败：{str(e)} !!!")
        raise  # 重新抛出异常以终止程序
    finally:
        if 'spark' in locals():
            spark.stop()
            print("Spark会话已关闭")


if __name__ == "__main__":
    main()