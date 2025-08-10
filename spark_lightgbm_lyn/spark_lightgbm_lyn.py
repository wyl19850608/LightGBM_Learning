from pyspark.sql.functions import pandas_udf, PandasUDFType, col, lit
from pyspark.sql.types import StructType, StructField, StringType, FloatType, DoubleType
import lightgbm as lgb
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
import argparse
import sys
import warnings
import pickle
import joblib

# 全局变量定义
global spark, dt8_1, dt8_2

# 解析输入日期参数
d = sys.argv[1]
print("d", d)
print("1" * 10)

for i in range(1):
    date_1 = datetime.strptime(d, "%Y%m%d") + timedelta(days=i)
    print("333-")
    dt8_1 = date_1.strftime("%Y%m%d")
    print("dt8_1:", dt8_1)
    dt8_2 = (date_1 - timedelta(days=1)).strftime("%Y%m%d")  # 修正日期格式符

# Spark配置参数
partition_mode = "--spark.sql.sources.partitionOverwriteMode"
compression = "--spark.sql.parquet.compression.codec"
print("3" * 10)

warnings.filterwarnings("ignore")

# 解析命令行参数
# parser = argparse.ArgumentParser(description="Process pySpark arguments.")
# parser.add_argument("--db_name", type=str, default="mrap_dev")
# parser.add_argument("--dt8_1", type=str, default=dt8_1)
# parser.add_argument("--dt8_2", type=str, default=dt8_2)
# parser.add_argument("--master", type=str, default="yarn")
# parser.add_argument("--spark.driver.memory", type=str, default="16g")  # 修正括号
# parser.add_argument("--spark.driver.memoryOverhead", type=str, default="6g")
# parser.add_argument("--spark.executor.memory", type=str, default="24g")  # 修正括号
# parser.add_argument("--spark.yarn.queue", type=str, default="root.tech.marketrd.memberrd.dev")
# parser.add_argument("--spark.executor.memoryOverhead", type=str, default="6g")  # 修正拼写和括号
# parser.add_argument("--spark.network.timeout", type=str, default="600s")  # 增加网络超时
# parser.add_argument("--spark.sql.shuffle.partitions", type=str, default="1500")  # 减少单个分区数据量
# parser.add_argument(partition_mode, type=str, default="dynamic")  # 修正引号
# parser.add_argument(compression, type=str, default="gzip")
#
# args, unknown = parser.parse_known_args()
#
# # 初始化Spark
# spark = findspark_msxf.init(
#     args=args,
#     envName="python3.8.5copy",
#     forcePackage=False,
#     appName="offline_predict_cust_level"
# )
# print("spark.version", spark.version)
# print("version:", findspark_msxf.__version__)  # 修正拼写

from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .appName("LightGBMGroupedPrediction") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .getOrCreate()

# 模型版本设置
now = datetime.now()
yesterday = now - timedelta(days=1)
model_version = yesterday.strftime("%Y%m%d")


class LightGBMSparkPredictor:
    def __init__(self, model_path):  # 修正构造函数名称
        """初始化LightGBM预测器
        :param model_path: LightGBM模型文件路径
        """
        self.model = self._load_model(model_path)
        self.feature_names = self.model.feature_name()
        self.model_params = self.model.params

    def _load_model(self, model_path):
        # 安全加载LightGBM模型
        if model_path.startswith(('hdfs://', 's3://')):  # 修正引号和括号
            # 从分布式存储加载
            print("name:", os.getpid())
            # local_path = f'./lgb_model_{os.getpid()}.pkl'
            local_path = f''
            os.system(f"hadoop fs -get {model_path} {local_path}")

            # 获取当前工作目录
            current_dir = os.getcwd()
            print(f"当前目录:{current_dir}")

            # 获取上一级目录
            alt_parent_dir = os.path.abspath(os.path.join(current_dir, ".."))  # 修正路径
            print(f"上级目录(替代方案):{alt_parent_dir}")

            # 打印目录内容
            print("\n目录内容:")
            for item in os.listdir(current_dir):
                print(f"-{item}")

            model_path = alt_parent_dir + "/model/" + "model_test_cp.pkl"
            print(f"尝试加载模型路径:{model_path}")
            local_model_path = "/home/finance/IDE/work/user_data/potential_custom_mining/feature_funcs/model/model_test_cp.pkl"  # 修正拼写和引号
            print(">>>1", local_model_path)

        else:
            # 从本地加载
            print(">>>2")
            model_path = "/home/finance/IDE/work/user_data/potential_custom_mining/feature_funcs/model/model_test_cp.txt"  # 修正拼写和引号
            model = lgb.Booster(model_file=model_path)  # .txt
            print("模型任务类型:", model.params.get("objective", "unknown"))  # 修正括号

        return model

    def _validate_features(self, input_columns):
        """验证输入特征与模型匹配"""
        if set(input_columns) != set(self.feature_names):
            missing = set(self.feature_names) - set(input_columns)
            extra = set(input_columns) - set(self.feature_names)
            raise ValueError(f"特征不匹配!\n缺少的特征:{missing}\n多余的特征:{extra}")

    def predict_spark_dataframe(self, spark_df):
        """
        对Spark DataFrame进行批量预测
        :param spark_df: 输入DataFrame,前两列为user_id和unique_id其余为特征列
        :return: 包含预测结果的DataFrame
        """
        # 获取特征列名(排除前两列ID列)
        input_columns = spark_df.columns[2:]
        # 验证特征匹配
        self._validate_features(input_columns)

        # 创建预测UDF
        @pandas_udf(self._get_output_schema(), PandasUDFType.GROUPED_MAP)
        def predict_batch(pdf):
            try:
                # 分离ID和特征
                ids = pdf[['user_id', 'unique_id']]
                X = pdf[self.feature_names]
                user_ids = pdf['user_id'].astype(str).tolist()
                unique_ids = pdf['unique_id'].astype(str).tolist()

                # 类型转换
                for col in X.columns:
                    if X[col].dtype == 'object':
                        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)  # 修正拼写
                    elif X[col].dtype == 'bool':
                        X[col] = X[col].astype(int)

                # 预测
                predictions = self.model.predict(X)
                print("predictions**** ", predictions, type(predictions))

                # 构造结果
                result = ids.copy()
                result['prediction'] = predictions

                # 如果是分类模型,添加概率
                if self._is_classification():
                    result['probability'] = predictions
                    if self._is_binary_classification():
                        result['prediction'] = (predictions > 0.5).astype(int)
                    elif self._is_multiclass_classification():
                        result['prediction'] = np.argmax(predictions, axis=1)
                        result['probability'] = np.max(predictions, axis=1)  # 修正拼写

                predictions = []
                for id in user_ids:
                    predictions.append(1.0)
                return result

            except Exception as e:
                # 错误处理
                print(f"预测出错:{str(e)}")
                error_df = pdf[['user_id', 'unique_id']].copy()
                error_df['prediction'] = np.nan
                if self._is_classification():
                    error_df['probability'] = np.nan
                return error_df

        # 执行预测(按分组避免单个任务过大)
        spark_df = spark_df.withColumn("group_key", col("user_id").substr(0, 10))
        result_df = spark_df.groupby("group_key").applyInPandas(  # 修正方法名
            predict_batch,
            schema=self._get_output_schema()
        )

        from pyspark.sql.functions import col, lit, when
        from pyspark.sql.types import StringType  # 修正空格

        ret_df = result_df
        ret_df = ret_df.drop('user_id')
        ret_df = ret_df.drop('unique_id')

        print("&" * 40)
        print("show schema:")
        ret_df.printSchema()  # 修正方法调用
        return ret_df

    def _get_output_schema(self):
        """定义输出Schema"""
        fields = [
            StructField("user_id", StringType(), True),
            StructField("unique_id", StringType(), True),
            StructField("prediction", DoubleType(), True)
        ]
        return StructType(fields)

    def _is_classification(self):
        """判断是否是分类模型"""
        return self.model_params.get('objective', '').startswith(('binary', 'multiclass'))

    def _is_binary_classification(self):
        return self.model_params.get('objective', '').startswith('binary')

    def _is_multiclass_classification(self):
        """判断是否是多分类模型"""
        return self.model_params.get('objective', '').startswith('multiclass')


def main():
    # 配置参数
    config = {
        "model_path": "/home/finance/IDE/work/user_data/potential_custom_mining/feature_funcs/model/model_test_cp.txt",  # 修正拼写
        "input_path": "/home/finance/IDE/work/user_data/potential_custom_mining/feature_funcs/train_data/coupon_v1_202250722_etl",  # 修正拼写
        "output_path": "hdfs://ms-dwh/user/yaning.liu/predice_yyh_t10_shenqing/coupon_v1",
        "sample_size": 5  # 显示样例数量
    }

    try:
        print(">>开始LightGBM批量预测流程<<<")
        start_time = time.time()

        # 1.创建预测器
        print("\n[步骤1/4]加载LightGBM模型...")
        predictor = LightGBMSparkPredictor(config["model_path"])
        print(f"模型加载完成,特征数:{len(predictor.feature_names)}")
        print(f"模型类型:{'分类' if predictor._is_classification() else '回归'}")  # 修正方法调用
        if predictor._is_classification():
            print(f"分类类型:{'二分类' if predictor._is_binary_classification() else '多分类'}")

        # 2.读取数据
        print("\n[步骤2/4]读取输入数据...")
        hdfs_path = 'hdfs://ms-dwh/user/yaning.liu/offline_feas/coupon_v1/20250723_etl/part-00003-1beb0a58-3842-4163-909d-643585eb0951-c00.csv'
        df = spark.read.csv(hdfs_path, header=True, inferSchema=True)
        columns_df = ['', ""]  # 修正变量名

        # 获取原始列名和类型
        internal_dtypes = df.dtypes
        df = df.limit(10)
        df = df.toDF(*columns_df)  # 使用新列名
        print(f"数据加载完成,总行数:{df.count()}")
        print("数据样例:")
        df.show(config["sample_size"], truncate=False)

        # 分类特征处理
        categorical_features = [
            'is_coupon_used', 'push_status', 'delay_type', 'tel_call_type',
            'touch_status', 'click_status', "is_coupon_issue", 'is_credit',
            'is_apply', "is_wdraw", "aprv_status", "is_realname_cert",
            'is_mobile_realname_cert', 't19_is_complete_base_info',
            't19_is_complete_occ_info', 't19_is_complete_contact_info',
            'reg_dev_compile_id', 'is_ms_sstaff', 'gender_cd',
            "residence_city_cd", "ocr_ethnic", "residence_pr_cd",
            'occ_cd', "census_pr_cd", 'census_city_cd', "census_area_cd",
            'residence_area_cd', 'cust_ty', 'cust_types_01', 'cust_types_02',
            'yls_cust_type', 'flagcd', 'kzr_types', 'elec_types', 'sms_types',
            'if_sms_yn', 'if_ele_yn', 'if_500up_yn', 'if_bj_10_yn',
            'if_bj_30_yn', 'is_login'
        ]  # 修正引号和拼写

        from pyspark.sql import functions as F
        cat_cols = [
            'db_src', 't19_user_status', 'user_mobile_status',
            'reg_dev_app_dl_chan_no', 'reg_chan_no', "reg_dev_info",
            'reg_dev_sys_type', 'reg_dev_sys_vn', 'reg_app_vn',
            'reg_dev_mfr', 'reg_dev_brand', 'reg_dev_model', 'reg_imei'
        ]  # 修正引号、拼写和标点

        print("label encoder")
        cat_cols = categorical_features + cat_cols
        for colname in cat_cols:
            df = df.withColumn(colname, F.col(colname).cast("string"))

        # 3.执行预测
        print("\n[步骤3/4]执行批量预测...")
        results = predictor.predict_spark_dataframe(df)
        print("&--before save" * 10)
        print("show schema:")
        results.printSchema()

        print("预测结果样例:")
        results.show(100)

        # 4.保存结果
        print("\n[步骤4/4]保存预测结果...")
        results.write.format("csv") \
            .option("header", "False") \
            .mode('overwrite') \
            .save(config["output_path"])
        print(f"预测结果已保存到:{config['output_path']}")

        # 性能统计
        elapsed = time.time() - start_time
        print(f"\n>>预测完成<<总耗时:{elapsed:.2f}秒")
        print(f"处理速度:{df.count()/elapsed:.2f}行/秒")

    except Exception as e:
        print(f"\n!!!预测流程出错:{str(e)}!!!!")
        raise e
    finally:
        spark.stop()


if __name__ == "__main__":
    main()