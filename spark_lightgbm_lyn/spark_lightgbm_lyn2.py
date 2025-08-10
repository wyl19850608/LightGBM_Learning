from pyspark.sql.functions import pandas_udf, PandasUDFType, col, lit, when, array_min, array_max, size, expr
from pyspark.sql.types import StructType, StructField, StringType, FloatType, DoubleType, IntegerType
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


def load_meta_data(meta_path):
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"元数据文件不存在: {meta_path}")
    meta_data = joblib.load(meta_path)
    return meta_data['cat_features'], meta_data['cat_info']

columns_df = [
    'user_id', 'unique_id', 'is_coupon_used', 'push_status', 'delay_type', 'delay_days', 'push_cnt', 'sms_charge_cnt',
    'tel_call_type', 'tel_call_dur', 'touch_status', 'click_status', 'is_coupon_issue', 'is_credit', 'is_apply',
    'is_wdraw', 'wdraw_cnt', 'wdraw_amt', 'wdraw_amt_t0', 'wdraw_amt_t3', 'wdraw_amt_t7', 'wdraw_amt_t10',
    'wdraw_amt_t14', 'wdraw_amt_t15', 'wdraw_amt_t30', 'tel_call_inav_cnt', 'subscribe_no', 'db_src', 'prod_cd',
    'touch_type', 'apply_amt', 'apply_cnt', 'aprv_status', 't19_user_status', 'user_mobile_status',
    'is_realname_cert', 'is_mobile_realname_cert', 'phy_del_op_side', 'reg_dev_app_dl_chan_no', 'reg_chan_no',
    'reg_attr_chan_no', 'chan_type', 'reg_dev_info', 'reg_dev_sys_type', 'reg_dev_sys_vn', 'reg_app_vn',
    'reg_dev_mfr', 'reg_dev_brand', 'reg_dev_model', 'reg_dev_compile_id', 'reg_imei', 't19is_complete_base_info',
    't19_is_complete_occ_info', 't19_is_complete_contact_info', 'is_ms_staff', 'age', 'gender_cd',
    'residence_city_cd', 'ocr_ethnic', 'residence_pr_cd', 'occ_cd', 'census_pr_cd', 'census_city_cd',
    'census_area_cd', 'residence_area_cd', 'cust_ty', 'ecust_types_01', 'cust_types_02', 'yls_cust_type',
    'last_fq_tx_amt', 'last_tx_amt', 'last_30d_login_app_days', 'last_30d_push_touch_times',
    'last_30d_sms_send_succ_cs', 'last_30d_tel_succ_cs', 'last_5d_coupon_cnt', 'last_10d_coupon_cnt',
    'querycncq03nwww', 'querycncq03wwlw', 'querycncq03wwpw', 'flagcd', 'cdaccapwwwwww', 'diaccapwwwwwww',
    'rvaccapwwww', 'rvrpsapwwwww', 'rvclmapwww', 'rvclaapwwwww', 'rewblsapwwww', 'rvnbmwpwwwww',
    'rvnbawpwwwww', 'rvapsapwwwwww', 'rvapmapwwww', 'rvapaapwwwww', 'pboc_repay_days', 'als_m1_id_bank_allnum',
    'als_mi_id_nbank_allnum', 'als_m3_id_bank_allnum', 'als_m3_id_nbank_allnum', 'query_times_bank_90d',
    'query_times_cfc_90d', 'last_30d_fq_cs', 'last_30d_tx_cs', 'last_30d_login_ayh_days', 'last_5d_lin_e_cnt',
    'last_10d_lin_e_ctnt', 'last_5d_gu_e_cnt', 'last_10d_gu_e_cnt', 'last_yunying_jj_new_rate',
    'last_yunying_jj_begin_time', 'last_yunying_jj_end_time', 'last_90d_yunying_jj_cs', 'last_180d_yunying_jj_cs',
    'partner_yx_contr_cnt', 'latest_login_days', 'total_loan_cnt', 'total_loan_amt', 'kzr_types', 'elec_types',
    'sms_types', 'if_sms_yn', 'if_ele_yn', 'if_500up_yn', 'if_bj_10_yn', 'if_bj_30_yn', 'avail_bal_cash',
    'avg_lim_use_rate', 'pril_bal', 'crdt_lim_yx', 'lim_use_rate', 'is_login'
]


class FeatureProcessor:
    def __init__(self, cat_info):
        self.cat_info = cat_info


    def process(self, spark_df):
        processed_df = spark_df
        processed_df = processed_df.toDF(*columns_df)  # 使用新列名

        result_df = processed_df.withColumn(
            "split_array",
            expr("filter(transform(split(pboc_repay_date_days, ',,'), x -> cast(trim(x) as int)), y -> y is not null)")
        ).withColumn(
            "min",
            when(col("split_array").isNotNull() & (size(col("split_array")) > 0), array_min(col("split_array")))
            .otherwise(None)
        ).withColumn(
            "max",
            when(col("split_array").isNotNull() & (size(col("split_array")) > 0), array_max(col("split_array")))
            .otherwise(None)
        ).withColumn(
            "mean",
            when(col("split_array").isNotNull() & (size(col("split_array")) > 0),
                 expr("round(aggregate(split_array, cast(0 as double), (acc, x) -> acc + x, acc -> acc / size(split_array)))"))
            .otherwise(None)
        ).select('pboc_repay_date_days', "min", "mean", "max")  # 只保留原始列和新增列

        processed_df = processed_df.join(result_df, on=["pboc_repay_date_days"], how="left")

        processed_df = processed_df.drop('pboc_repay_date_days')
        print("after drop final_df.columns", len(processed_df.columns))

        for feat in self.cat_info.keys():
            valid_cats = self.cat_info[feat]['original']
            processed_df = processed_df.withColumn(
                feat,
                when(col(feat).isin(valid_cats), col(feat)).otherwise(lit('unknown'))
            )
        return processed_df

class LightGBMPredictor:
    def __init__(self, model_path, cat_features, cat_info):  # 修正构造函数名称
        """初始化LightGBM预测器
        :param model_path: LightGBM模型文件路径
        """
        self.model = lgb.Booster(model_file=model_path)
        self.feature_names = self.model.feature_name()
        self.cat_features = cat_features  # 使用加载的分类特征列表
        self.num_features = [f for f in self.feature_names if f not in self.cat_features]
        self.cat_info = cat_info
        self.output_schema = StructType([
            StructField("user_id", StringType(), True),
            StructField("unique_id", StringType(), True),
            StructField("prediction", DoubleType(), True)
        ])

    def predict(self, spark_df):
        # 广播模型和分类信息
        model_broadcast = spark_df._sc.broadcast(self.model)
        cat_info_broadcast = spark_df._sc.broadcast(self.cat_info)
        cat_features_broadcast = spark_df._sc.broadcast(self.cat_features)
        feature_names = self.feature_names
        num_features = self.num_features

        def predict_batch(pdf):
            """处理单批数据并返回DataFrame"""
            model = model_broadcast.value
            cat_info = cat_info_broadcast.value
            cat_features = cat_features_broadcast.value

            if len(pdf) == 0:
                return pd.DataFrame(columns=['id', 'prediction', 'probability'])

            # 提取ID
            ids = pdf['id'].tolist()

            # 处理特征
            features = {f: [] for f in feature_names}

            # 处理分类特征
            for feat in cat_features:
                cat_map = {cat: idx for idx, cat in enumerate(cat_info[feat]['original'])}
                cat_map['unknown'] = -1
                for val in pdf[feat].tolist():
                    features[feat].append(cat_map.get(val, -1))

            # 处理数值特征
            for feat in num_features:
                for val in pdf[feat].tolist():
                    if pd.isna(val):
                        features[feat].append(0.0)
                    else:
                        features[feat].append(float(val))

            # 创建特征DataFrame
            X = pd.DataFrame(features)

            # 执行预测
            probabilities = model.predict(X).tolist()
            predictions = [int(p > 0.5) for p in probabilities]

            # 构造结果
            return pd.DataFrame({
                'id': ids,
                'prediction': predictions,
                'probability': probabilities
            })

        # 分组预测
        spark_df = spark_df.withColumn("group_key", col("id").substr(0, 2))
        result_df = spark_df.groupby("group_key").applyInPandas(
            predict_batch,
            schema=self.output_schema
        )

        return result_df.drop("group_key")

def main():
    # 配置参数
    config = {
        "model_path": "/home/finance/IDE/work/user_data/potential_custom_mining/feature_funcs/model/model_test_cp.txt",  # 修正拼写
        "input_path": "/home/finance/IDE/work/user_data/potential_custom_mining/feature_funcs/train_data/coupon_v1_202250722_etl",  # 修正拼写
        "output_path": "hdfs://ms-dwh/user/yaning.liu/predice_yyh_t10_shenqing/coupon_v1",
        "sample_size": 5  # 显示样例数量
    }
    meta_path = "cat_meta_data.pkl"

    try:
        print(">>开始LightGBM批量预测流程<<<")
        start_time = time.time()


        print("加载分类特征信息...")
        loaded_cat_features, loaded_cat_info = load_meta_data(meta_path)

        # 使用加载的分类特征列表初始化预测器
        predictor = LightGBMPredictor(config["model_path"], loaded_cat_features, loaded_cat_info)

        # 1.创建预测器
        print(f"模型加载完成,特征数:{len(predictor.feature_names)}")

        # 2.读取数据
        print("\n[步骤2/4]读取输入数据...")
        hdfs_path = 'data.csv'
        df = spark.read.csv(hdfs_path, header=True, inferSchema=True)
        columns_df = ['', ""]  # 修正变量名

        print("预处理测试数据...")
        processor = FeatureProcessor(loaded_cat_info)
        processed_df = processor.process(df)

        # 获取原始列名和类型
        internal_dtypes = df.dtypes
        df = df.limit(10)
        df = df.toDF(*columns_df)  # 使用新列名
        print(f"数据加载完成,总行数:{df.count()}")
        print("数据样例:")
        df.show(config["sample_size"], truncate=False)

        print("执行预测...")
        result_df = predictor.predict(processed_df)

        print("预测结果示例:")
        result_df.show(10)

        result_df.write.mode("overwrite").csv("prediction_results", header=True)
        print("预测结果已保存至: prediction_results")
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