import os
import time
import warnings
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, size, expr, array_min, array_max, count

from config import DATA_CONFIG, SPARK_CONFIG, DATA_GEN_CONFIG, MODEL_CONFIG

# 配置与常量定义
warnings.filterwarnings("ignore")

# 所有列定义
COLUMNS_DF = [
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
print(len(COLUMNS_DF))
exit()

# 明确指定分类变量及其可能的取值范围
CATEGORICAL_COLUMNS = {
    'is_coupon_used': ['0', '1'],
    'push_status': ['success', 'failed', 'pending'],
    'delay_type': ['type1', 'type2', 'type3', 'type4'],
    'tel_call_type': ['inbound', 'outbound', 'missed'],
    'touch_status': ['touched', 'untouched'],
    'click_status': ['clicked', 'not_clicked'],
    'is_coupon_issue': ['0', '1'],
    'is_credit': ['0', '1'],
    'is_apply': ['0', '1'],
    'is_wdraw': ['0', '1'],
    'aprv_status': ['approved', 'rejected', 'pending'],
    'is_realname_cert': ['0', '1'],
    'is_mobile_realname_cert': ['0', '1'],
    't19is_complete_base_info': ['0', '1'],
    't19_is_complete_occ_info': ['0', '1'],
    't19_is_complete_contact_info': ['0', '1'],
    'reg_dev_compile_id': ['id1', 'id2', 'id3', 'id4', 'id5'],
    'is_ms_staff': ['0', '1'],
    'gender_cd': ['M', 'F', 'U'],
    'residence_city_cd': ['C001', 'C002', 'C003', 'C004', 'C005'],
    'ocr_ethnic': ['HAN', 'MIN', 'ZHUANG', 'YI', 'OTHER'],
    'residence_pr_cd': ['P001', 'P002', 'P003', 'P004'],
    'occ_cd': ['O001', 'O002', 'O003', 'O004', 'O005'],
    'census_pr_cd': ['P001', 'P002', 'P003', 'P004'],
    'census_city_cd': ['C001', 'C002', 'C003', 'C004', 'C005'],
    'census_area_cd': ['A001', 'A002', 'A003', 'A004'],
    'residence_area_cd': ['A001', 'A002', 'A003', 'A004'],
    'cust_ty': ['T01', 'T02', 'T03', 'T04'],
    'ecust_types_01': ['ET01', 'ET02', 'ET03'],
    'cust_types_02': ['CT01', 'CT02', 'CT03', 'CT04'],
    'yls_cust_type': ['Y01', 'Y02', 'Y03'],
    'flagcd': ['F01', 'F02', 'F03', 'F04', 'F05'],
    'kzr_types': ['K01', 'K02', 'K03'],
    'elec_types': ['E01', 'E02', 'E03', 'E04'],
    'sms_types': ['S01', 'S02', 'S03'],
    'if_sms_yn': ['0', '1'],
    'if_ele_yn': ['0', '1'],
    'if_500up_yn': ['0', '1'],
    'if_bj_10_yn': ['0', '1'],
    'if_bj_30_yn': ['0', '1'],
    'is_login': ['0', '1'],
    'db_src': ['src1', 'src2', 'src3'],
    't19_user_status': ['active', 'inactive', 'suspended'],
    'user_mobile_status': ['valid', 'invalid', 'expired'],
    'reg_dev_app_dl_chan_no': ['CH01', 'CH02', 'CH03', 'CH04'],
    'reg_chan_no': ['CH01', 'CH02', 'CH03', 'CH04'],
    'reg_dev_info': ['INFO1', 'INFO2', 'INFO3'],
    'reg_dev_sys_type': ['Android', 'iOS', 'Other'],
    'reg_dev_sys_vn': ['V1', 'V2', 'V3', 'V4'],
    'reg_app_vn': ['V1.0', 'V2.0', 'V3.0'],
    'reg_dev_mfr': ['MFR1', 'MFR2', 'MFR3', 'MFR4'],
    'reg_dev_brand': ['BR1', 'BR2', 'BR3', 'BR4', 'BR5'],
    'reg_dev_model': ['MOD1', 'MOD2', 'MOD3', 'MOD4'],
    'reg_imei': ['IM1', 'IM2', 'IM3', 'IM4']
}

# UUID列（唯一标识符）
UUID_COLUMNS = ['user_id', 'unique_id']

# 标签列
LABEL_COLUMN = 'label'

# 自动计算数值列（所有不在分类列、UUID列和标签列中的列）
NUMERIC_COLUMNS = [col for col in COLUMNS_DF
                   if col not in CATEGORICAL_COLUMNS.keys()
                   and col not in UUID_COLUMNS
                   and col != LABEL_COLUMN]

# 特征相关常量
REQUIRED_COLUMNS = ["pboc_repay_days", "label"]  # 训练数据需要标签列
DEFAULT_NULL_PROBABILITY = 0.1  # 全局默认空值概率（10%）
# 针对特定列的空值概率调整
COLUMN_NULL_PROBABILITIES = {
    'is_coupon_used': 0.05,
    'push_status': 0.05,
    'gender_cd': 0.05,
    'age': 0.15,
    'delay_days': 0.15,
    'wdraw_amt': 0.15,
    'label': 0.0,
    'pboc_repay_days': 0.05
}


class FeatureProcessor:
    """特征处理器：控制空值比例，确保数据可用于训练"""

    def __init__(self, cat_info=None):
        self.cat_info = cat_info if cat_info is not None else {}
        self.is_fitted = False  # 标记是否已拟合
        self.all_columns = COLUMNS_DF
        self.categorical_columns = list(CATEGORICAL_COLUMNS.keys())
        self.numeric_columns = NUMERIC_COLUMNS
        self.uuid_columns = UUID_COLUMNS
        self.label_column = LABEL_COLUMN
        self.category_values = CATEGORICAL_COLUMNS  # 存储分类变量的可能取值

    def fit(self, spark_df):
        """拟合处理器：从训练数据中验证分类特征的有效类别"""
        spark_df = self._ensure_all_columns(spark_df)
        self._check_required_columns(spark_df)
        self._check_null_ratios(spark_df)

        # 收集每个分类特征的有效类别（包含空值）
        for feature in self.categorical_columns:
            if feature in spark_df.columns:
                categories = spark_df.select(feature).distinct().rdd.flatMap(lambda x: x).collect()
                str_categories = [str(cat) if cat is not None else None for cat in categories]
                self.cat_info[feature] = {
                    "original": str_categories,
                    "count": len(str_categories),
                    "null_count": spark_df.filter(col(feature).isNull()).count(),
                    "null_ratio": spark_df.filter(col(feature).isNull()).count() / spark_df.count()
                }

        self.is_fitted = True
        return self

    def transform(self, spark_df):
        """转换数据：生成衍生特征，保持合理空值比例"""
        if not self.is_fitted and self.cat_info is None:
            raise RuntimeError("特征处理器尚未拟合，请先调用fit方法")

        spark_df = self._ensure_all_columns(spark_df)
        self._check_required_columns(spark_df)
        processed_df = self._derive_date_stats(spark_df)
        return processed_df

    def fit_transform(self, spark_df):
        """拟合并转换数据"""
        return self.fit(spark_df).transform(spark_df)

    def _check_required_columns(self, spark_df):
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in spark_df.columns]
        if missing_cols:
            raise ValueError(f"输入数据缺少必要列: {missing_cols}")

    def _check_null_ratios(self, spark_df):
        """检查各列空值比例，确保不会过高"""
        total_rows = spark_df.count()
        if total_rows == 0:
            return

        print("\n检查空值比例:")
        for col_name in spark_df.columns:
            null_count = spark_df.filter(col(col_name).isNull()).count()
            null_ratio = null_count / total_rows if total_rows > 0 else 0
            max_allowed = COLUMN_NULL_PROBABILITIES.get(col_name, DEFAULT_NULL_PROBABILITY) * 1.5
            if null_ratio > max_allowed:
                print(f"警告: 列 {col_name} 空值比例过高 ({null_ratio:.2%})")
            elif null_ratio > 0:
                print(f"列 {col_name} 空值比例: {null_ratio:.2%}")

    def _ensure_all_columns(self, spark_df):
        """确保所有必要的列都存在，不存在的列根据类型初始化"""
        existing_cols = spark_df.columns

        # 处理UUID列
        for col_name in self.uuid_columns:
            if col_name not in existing_cols:
                spark_df = spark_df.withColumn(col_name, lit(0).cast("integer"))

        # 处理标签列
        if self.label_column not in existing_cols:
            spark_df = spark_df.withColumn(self.label_column, lit(0).cast("integer"))

        # 处理分类列
        for col_name in self.categorical_columns:
            if col_name not in existing_cols and col_name in self.category_values:
                default_value = self.category_values[col_name][0] if self.category_values[col_name] else None
                spark_df = spark_df.withColumn(col_name, lit(default_value).cast("string"))

        # 处理数值列
        for col_name in self.numeric_columns:
            if col_name not in existing_cols:
                spark_df = spark_df.withColumn(col_name, lit(0.0).cast("double"))

        return spark_df

    def _derive_date_stats(self, spark_df):
        # 从pboc_repay_days衍生统计特征
        date_stats_df = spark_df.withColumn(
            "split_array",
            expr("filter(transform(split(pboc_repay_days, ',,'), x -> cast(trim(x) as int)), y -> y is not null)")
        ).withColumn(
            "min_repay_day", array_min(col("split_array")).alias("min_repay_day")
        ).withColumn(
            "max_repay_day", array_max(col("split_array")).alias("max_repay_day")
        ).withColumn(
            "mean_repay_day",
            expr("round(aggregate(split_array, cast(0 as double), "
                 "(acc, x) -> acc + x, acc -> acc / size(split_array)))").alias("mean_repay_day")
        ).select("user_id", "min_repay_day", "mean_repay_day", "max_repay_day")

        result_df = spark_df.join(date_stats_df, on="user_id", how="left")
        return result_df


def save_metadata(processor, output_path):
    """保存特征元数据供预测时使用"""
    if not processor.is_fitted:
        raise RuntimeError("无法保存未拟合的处理器元数据")

    meta_data = {
        "cat_features": processor.categorical_columns,
        "numeric_features": processor.numeric_columns,
        "uuid_features": processor.uuid_columns,
        "label_column": processor.label_column,
        "date_features": ["pboc_repay_days"],
        "all_columns": processor.all_columns,
        "cat_info": processor.cat_info,
        "required_columns": REQUIRED_COLUMNS,
        "null_probabilities": COLUMN_NULL_PROBABILITIES,
        "generated_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(meta_data, output_path)
    print(f"元数据已保存至：{output_path}")


def create_sample_raw_data(file_path, sample_size=1000):
    """创建控制空值比例的示例原始数据"""
    print(f"创建示例原始数据到 {file_path}")
    np.random.seed(42)
    data = {}

    # 生成UUID列
    data['user_id'] = range(1, sample_size + 1)
    data['unique_id'] = range(1001, 1001 + sample_size)

    # 生成标签列
    data['label'] = np.random.randint(0, 2, sample_size)

    # 生成分类列
    for col_name, values in CATEGORICAL_COLUMNS.items():
        null_prob = COLUMN_NULL_PROBABILITIES.get(col_name, DEFAULT_NULL_PROBABILITY)
        col_data = np.random.choice(values, sample_size)
        mask = np.random.random(sample_size) < null_prob
        col_data[mask] = np.nan
        data[col_name] = col_data

    # 生成数值列
    for col_name in NUMERIC_COLUMNS:
        null_prob = COLUMN_NULL_PROBABILITIES.get(col_name, DEFAULT_NULL_PROBABILITY)
        if col_name == 'age':
            col_data = np.random.randint(18, 71, sample_size).astype(float)
        elif 'cnt' in col_name or 'days' in col_name or 'times' in col_name:
            col_data = np.random.randint(0, 100, sample_size).astype(float)
        elif 'amt' in col_name or 'rate' in col_name:
            col_data = np.abs(np.random.randn(sample_size) * 1000 + 5000)
        else:
            col_data = np.random.randn(sample_size) * 10 + 50

        mask = np.random.random(sample_size) < null_prob
        col_data[mask] = np.nan
        data[col_name] = col_data

    # 特殊处理pboc_repay_days列
    null_prob = COLUMN_NULL_PROBABILITIES.get('pboc_repay_days', DEFAULT_NULL_PROBABILITY)
    pboc_data = []
    for _ in range(sample_size):
        if np.random.random() < null_prob:
            pboc_data.append(np.nan)
        else:
            num_days = np.random.randint(1, 6)
            days = np.random.randint(1, 31, num_days)
            pboc_data.append(",,".join(map(str, days)))
    data['pboc_repay_days'] = pboc_data

    # 保存数据
    df = pd.DataFrame(data)[COLUMNS_DF]
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    total_values = df.size
    null_values = df.isnull().sum().sum()
    print(f"生成的示例数据整体空值比例: {null_values / total_values:.2%}")


def analyze_null_distribution(spark_df):
    """分析数据中的空值分布"""
    if spark_df.count() == 0:
        print("数据为空，无法分析空值分布")
        return

    print("\n空值分布分析:")
    total_rows = spark_df.count()
    null_counts = {}
    for col_name in spark_df.columns:
        null_count = spark_df.filter(col(col_name).isNull()).count()
        null_ratio = null_count / total_rows if total_rows > 0 else 0
        null_counts[col_name] = (null_count, null_ratio)

    # 按空值比例排序并显示前10名
    sorted_cols = sorted(null_counts.items(), key=lambda x: x[1][1], reverse=True)
    for col_name, (count, ratio) in sorted_cols[:10]:
        print(f"{col_name}: {count}个空值 ({ratio:.2%})")


def main():
    try:
        start_time = time.time()
        print(">>> 开始训练数据生成流程 <<<")
        print(f"总列数: {len(COLUMNS_DF)}, 分类列: {len(CATEGORICAL_COLUMNS)}, 数值列: {len(NUMERIC_COLUMNS)}")

        # 创建示例数据（如果不存在）
        if not os.path.exists(DATA_CONFIG["raw_train_data"]):
            print(f"创建示例数据到 {DATA_CONFIG['raw_train_data']}")
            create_sample_raw_data(
                DATA_CONFIG["raw_train_data"],
                DATA_GEN_CONFIG["generate_sample_size"]
            )

        # 初始化Spark
        spark = (SparkSession.builder
                 .appName(SPARK_CONFIG["appName"])
                 .master(SPARK_CONFIG["master"])
                 .config("spark.driver.memory", SPARK_CONFIG["driver_memory"])
                 .config("spark.executor.memory", SPARK_CONFIG["executor_memory"])
                 .config("spark.sql.execution.arrow.pyspark.enabled", SPARK_CONFIG["arrow_enabled"])
                 .getOrCreate())
        print(f"Spark初始化完成（版本：{spark.version}）")

        # 读取原始数据
        print(f"读取原始训练数据（路径：{DATA_CONFIG['raw_train_data']}）...")
        raw_df = spark.read.csv(
            DATA_CONFIG["raw_train_data"],
            header=True,
            inferSchema=True,
            quote='"',
            escape='"'
        )
        print(f"原始数据加载完成（行数：{raw_df.count()}，原始列数：{len(raw_df.columns)}）")

        # 分析空值分布
        analyze_null_distribution(raw_df)

        # 特征处理
        print("\n开始特征处理...")
        feature_processor = FeatureProcessor()
        processed_df = feature_processor.fit_transform(raw_df)
        print(f"特征处理完成（处理后列数：{len(processed_df.columns)}）")

        # 保存处理后的数据
        print(f"保存处理后的训练数据至：{DATA_CONFIG['processed_train_data']}")
        os.makedirs(DATA_CONFIG["processed_train_data"], exist_ok=True)
        processed_df.write.mode("overwrite").csv(
            DATA_CONFIG["processed_train_data"],
            header=True,
            quote='"',
            escape='"'
        )

        # 保存元数据
        save_metadata(feature_processor, MODEL_CONFIG["meta_data"])

        # 展示样例
        print(f"\n处理后的数据样例（前{DATA_GEN_CONFIG['sample_size']}行）：")
        show_columns = ["user_id", "unique_id", "label", "is_coupon_used", "push_status",
                        "gender_cd", "age", "delay_days", "min_repay_day", "mean_repay_day", "max_repay_day"]
        processed_df.select(show_columns).show(DATA_GEN_CONFIG["sample_size"], truncate=False)

        # 统计信息
        elapsed = time.time() - start_time
        print(f"\n>>> 训练数据生成流程完成 <<<")
        print(f"总耗时：{elapsed:.2f}秒")
        print(f"处理速度：{raw_df.count() / elapsed:.2f}行/秒")

    except Exception as e:
        print(f"\n!!! 训练数据生成失败：{str(e)} !!!")
        raise
    finally:
        if 'spark' in locals():
            spark.stop()
            print("Spark会话已关闭")


if __name__ == "__main__":
    main()