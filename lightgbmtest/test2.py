from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType
from pyspark.sql.functions import col, when, lit, spark_partition_id
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
import os
import shutil
import joblib

# 定义所有特征列 - 与generate_training_data.py保持一致
COLUMNS_DF = [
    'user_id', 'unique_id', 'is_coupon_used', 'push_status', 'delay_type', 'delay_days', 'push_cnt', 'sms_charge_cnt',
    'tel_call_type', 'tel_call_dur', 'touch_status', 'click_status', 'is_coupon_issue', 'is_credit', 'is_apply',
    'is_wdraw', 'wdraw_cnt', 'wdraw_amt', 'wdraw_amt_t0', 'wdraw_amt_t3', 'wdraw_amt_t7', 'wdraw_amt_t10',
    'wdraw_amt_t14', 'wdraw_amt_t15', 'wdraw_amt_t30', 'tel_call_inav_cnt', 'subscribe_no', 'db_src', 'prod_cd',
    'touch_type', 'apply_amt', 'apply_cnt', 'aprv_status', 't19_user_status', 'user_mobile_status',
    'is_realname_cert', 'is_mobile_realname_cert', 'phy_del_op_side', 'reg_dev_app_dl_chan_no', 'reg_chan_no',
    'reg_attr_chan_no', 'chan_type', 'reg_dev_info', 'reg_dev_sys_type', 'reg_dev_sys_vn', 'reg_app_vn',
    'reg_dev_mfr', 'reg_dev_brand', 'reg_dev_model', 'reg_dev_compile_id', 'reg_imei', 't19_is_complete_base_info',
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

# 定义分类特征 - 与generate_training_data.py保持一致
categorical_features = [
    'is_coupon_used', 'push_status', 'delay_type', 'tel_call_type',
    'touch_status', 'click_status', "is_coupon_issue", 'is_credit',
    'is_apply', "is_wdraw", "aprv_status", "is_realname_cert",
    'is_mobile_realname_cert', 't19_is_complete_base_info',
    't19_is_complete_occ_info', 't19_is_complete_contact_info',
    'reg_dev_compile_id', 'is_ms_staff', 'gender_cd',
    "residence_city_cd", "ocr_ethnic", "residence_pr_cd",
    'occ_cd', "census_pr_cd", 'census_city_cd', "census_area_cd",
    'residence_area_cd', 'cust_ty', 'ecust_types_01', 'cust_types_02',
    'yls_cust_type', 'flagcd', 'kzr_types', 'elec_types', 'sms_types',
    'if_sms_yn', 'if_ele_yn', 'if_500up_yn', 'if_bj_10_yn',
    'if_bj_30_yn', 'is_login', 'db_src', 't19_user_status', 'user_mobile_status',
    'reg_dev_app_dl_chan_no', 'reg_chan_no', "reg_dev_info",
    'reg_dev_sys_type', 'reg_dev_sys_vn', 'reg_app_vn',
    'reg_dev_mfr', 'reg_dev_brand', 'reg_dev_model', 'reg_imei'
]

# 连续特征是总特征减去分类特征
continuous_features = [col for col in COLUMNS_DF if col not in categorical_features]

# 初始化SparkSession
def init_spark():
    return SparkSession.builder \
        .appName("LightGBMWithCustomFeatures") \
        .master("local[*]") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .getOrCreate()

# 加载由generate_training_data.py生成的训练数据
def load_training_data(spark, file_path="training_data.csv"):
    """加载生成的训练数据并添加标签列"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"训练数据文件不存在: {file_path}，请先运行generate_training_data.py生成数据")


    schema = StructType([
        StructField(col_name, StringType(), nullable=True)
        for col_name in COLUMNS_DF
    ])

    # 加载数据
    df = spark.read.csv(
        file_path,
        header=False,               # 包含表头
        inferSchema=False,         # 不自动推断类型（关键：避免自动转为数值型）
        encoding='utf-8',          # 编码格式
        nullValue='',              # 将空字符串视为 null
        quote='"',                 # 引号字符
        escape='"',                # 转义字符
        multiLine=False            # 是否允许多行记录（根据实际数据调整）
    )

    # 确保列顺序正确
    df = df.toDF(*COLUMNS_DF)



    print(f"成功加载训练数据，共{df.count()}行，{len(df.columns)}列特征")
    return df

# 训练模型 - 修改为使用加载的特征数据
def train_lightgbm_model(spark, model_path, meta_path, data_path="training_data.csv"):
    # 加载自定义特征数据
    df = load_training_data(spark, data_path)
    # 转换为pandas DataFrame进行处理
    pdf = df.toPandas()
    # 为演示目的，基于某些特征生成标签（实际应用中应该使用真实标签）
    # 这里使用一个简单的规则生成标签：如果优惠券被使用且有提现金额，则标签为1

    for feat in continuous_features:
        # 空字符串转为NaN，再填充-1
        pdf[feat] = pd.to_numeric(pdf[feat], errors='coerce').fillna(-1.0)

    # 处理用于生成标签的特征
    pdf['is_coupon_used'] = pdf['is_coupon_used'].fillna('0')
    pdf['wdraw_amt'] = pd.to_numeric(pdf['wdraw_amt'], errors='coerce').fillna(0.0)
    pdf['label'] = ((pdf['is_coupon_used'] == "1") & (pdf['wdraw_amt'] > 0)).astype(int)
    print(f"正样本比例: {pdf['label'].mean():.4f}")

    # 使用预定义的特征分类
    cat_features = categorical_features
    num_features = continuous_features
    feature_names = cat_features + num_features

    # 对分类特征进行编码
    for feat in cat_features:
        # 确保特征是字符串类型再进行编码
        pdf[feat] = pdf[feat].astype(str).astype('category').cat.codes

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        pdf[feature_names], pdf['label'], test_size=0.2, random_state=42, stratify=pdf['label']
    )


    # 移除不需要的ID列
    drop_cols = ['user_id', 'unique_id']
    X_train = X_train.drop(columns=[col for col in drop_cols if col in X_train.columns])
    X_val = X_val.drop(columns=[col for col in drop_cols if col in X_val.columns])


    print("X_train特征数:", len(X_train.columns))
    print("原始数据特征数:", len(pdf.columns))



    # 构建LightGBM数据集
    lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=[i for i, f in enumerate(X_train.columns) if f in cat_features])
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train, categorical_feature=[i for i, f in enumerate(X_train.columns) if f in cat_features])

    # 训练参数 - 根据实际问题调整
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': 1,
        'seed': 42,
        'class_weight': 'balanced'  # 处理可能的类别不平衡
    }

    # 训练模型
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        num_boost_round=100,
    )

    # 保存模型
    if os.path.exists(model_path):
        if os.path.isfile(model_path):
            os.remove(model_path)
        else:
            shutil.rmtree(model_path, ignore_errors=True)
    model.save_model(model_path)
    print(f"模型已保存至: {model_path}")

    # 生成并保存分类特征信息
    cat_info = {}
    # 重新加载原始数据以获取未编码的类别信息
    original_df = pd.read_csv(data_path, encoding='utf-8', names=COLUMNS_DF, header=None)
    for feat in cat_features:
        if feat not in original_df.columns:
            continue
        # 确保特征是字符串类型
        original_df[feat] = original_df[feat].astype(str)
        original_cats = original_df[feat].unique().tolist()
        cat_info[feat] = {
            'original': original_cats,
            'encoded': list(range(len(original_cats))),
            'num_categories': len(original_cats)
        }

    # 保存所有特征信息
    meta_data = {
        'cat_features': cat_features,
        'num_features': num_features,
        'all_features': feature_names,
        'cat_info': cat_info
    }
    joblib.dump(meta_data, meta_path)
    print(f"特征元数据已保存至: {meta_path}")

    return cat_features, num_features, cat_info

# 加载特征信息
def load_meta_data(meta_path):
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"元数据文件不存在: {meta_path}")
    meta_data = joblib.load(meta_path)
    return (meta_data['cat_features'],
            meta_data['num_features'],
            meta_data['all_features'],
            meta_data['cat_info'])

# 生成测试数据 - 基于自定义特征结构
def generate_test_data(spark, all_features, cat_features, num_features, sample_size=20000):
    np.random.seed(43)
    data = {
        "id": [f"test_{i:08d}" for i in range(sample_size)],
    }

    # 生成数值特征
    for feat in num_features:
        if feat in ['user_id', 'unique_id', 'subscribe_no', 'prod_cd']:
            data[feat] = [int(np.random.randint(10000000, 99999999)) for _ in range(sample_size)]
        elif feat in ['delay_days', 'push_cnt', 'sms_charge_cnt', 'tel_call_dur',
                      'wdraw_cnt', 'tel_call_inav_cnt', 'apply_cnt', 'age']:
            data[feat] = [int(np.random.randint(0, 100)) for _ in range(sample_size)]
        elif 'amt' in feat or 'amount' in feat:  # 金额类特征
            data[feat] = [float(np.round(np.random.uniform(0, 100000), 2)) for _ in range(sample_size)]
        elif 'rate' in feat:  # 比率类特征
            data[feat] = [float(np.round(np.random.uniform(0, 100), 4)) for _ in range(sample_size)]
        elif 'time' in feat or 'date' in feat:  # 时间类特征
            from faker import Faker
            fake = Faker('zh_CN')
            data[feat] = [fake.date_between(start_date='-1y', end_date='today').isoformat() for _ in range(sample_size)]
        else:  # 其他数值特征
            data[feat] = [float(np.round(np.random.uniform(0, 100), 2)) for _ in range(sample_size)]

    # 生成分类特征
    for feat in cat_features:
        if feat in ['is_coupon_used', 'is_credit', 'is_apply', 'is_wdraw',
                    'is_realname_cert', 'is_mobile_realname_cert', 'is_ms_staff',
                    'if_sms_yn', 'if_ele_yn', 'if_500up_yn', 'if_bj_10_yn', 'if_bj_30_yn', 'is_login']:
            # 布尔型特征
            data[feat] = [str(np.random.choice([0, 1])) for _ in range(sample_size)]
        elif feat == 'gender_cd':
            # 性别
            data[feat] = [np.random.choice(['男', '女', '未知']) for _ in range(sample_size)]
        elif feat == 'ocr_ethnic':
            # 民族
            ethnicities = ['汉', '蒙古', '回', '藏', '维吾尔', '苗', '彝', '壮', '布依', '朝鲜']
            data[feat] = [np.random.choice(ethnicities) for _ in range(sample_size)]
        elif feat in ['reg_dev_compile_id', 'reg_imei']:
            # 设备唯一标识
            from faker import Faker
            fake = Faker('zh_CN')
            data[feat] = [fake.uuid4()[:16] for _ in range(sample_size)]
        elif feat in ['db_src', 'reg_dev_app_dl_chan_no', 'reg_chan_no', 'reg_dev_mfr', 'reg_dev_brand']:
            # 渠道或品牌编码
            data[feat] = [f"code_{np.random.randint(100, 999)}" for _ in range(sample_size)]
        else:  # 其他分类特征
            data[feat] = [f"cat_{np.random.randint(0, 5)}" for _ in range(sample_size)]

    # 创建DataFrame并确保类型正确
    pdf = pd.DataFrame(data)

    # 定义Schema
    schema_fields = [StructField("id", StringType(), True)]
    for feat in num_features:
        if 'time' in feat or 'date' in feat:
            schema_fields.append(StructField(feat, DateType(), True))
        elif feat in ['user_id', 'unique_id', 'subscribe_no', 'prod_cd',
                      'delay_days', 'push_cnt', 'age']:  # 整数型数值特征
            schema_fields.append(StructField(feat, IntegerType(), True))
        else:  # 其他数值特征
            schema_fields.append(StructField(feat, DoubleType(), True))

    for feat in cat_features:
        schema_fields.append(StructField(feat, StringType(), True))

    schema = StructType(schema_fields)

    return spark.createDataFrame(pdf, schema=schema)

# 特征预处理
class FeatureProcessor:
    def __init__(self, cat_info, cat_features, num_features):
        self.cat_info = cat_info
        self.cat_features = cat_features
        self.num_features = num_features

    def check_features(self, spark_df):
        """检查DataFrame是否包含所有必要的特征"""
        df_columns = set(spark_df.columns)
        required_features = set(self.cat_features + self.num_features)

        missing = required_features - df_columns
        if missing:
            raise ValueError(f"缺少必要的特征: {missing}")

        extra = df_columns - required_features - {'id'}
        if extra:
            print(f"警告: 存在未使用的额外特征: {extra}")

        return True

    def process(self, spark_df):
        # 先检查特征完整性
        self.check_features(spark_df)

        processed_df = spark_df
        # 处理分类特征
        for feat in self.cat_features:
            valid_cats = [str(cat) for cat in self.cat_info[feat]['original']]  # 确保是字符串比较
            processed_df = processed_df.withColumn(
                feat,
                when(col(feat).isin(valid_cats), col(feat)).otherwise(lit('unknown'))
            )

        # 处理数值特征缺失值
        for feat in self.num_features:
            processed_df = processed_df.withColumn(
                feat,
                when(col(feat).isNull(), lit(0.0)).otherwise(col(feat))
            )

        return processed_df

# LightGBM预测器
class LightGBMPredictor:
    def __init__(self, model_path, cat_features, num_features, all_features, cat_info):
        self.model = lgb.Booster(model_file=model_path)
        self.feature_names = all_features
        self.cat_features = cat_features
        self.num_features = num_features
        self.cat_info = cat_info
        self.output_schema = StructType([
            StructField("id", StringType(), True),
            StructField("prediction", IntegerType(), True),
            StructField("probability", DoubleType(), True)
        ])

        # 验证模型特征与元数据特征一致性
        model_feats = set(self.model.feature_name())
        meta_feats = set(all_features)
        meta_feats.discard('user_id')
        meta_feats.discard('unique_id')
        if model_feats != meta_feats:
            raise ValueError(f"模型特征与元数据特征不一致: 模型有{model_feats - meta_feats}, 元数据有{meta_feats - model_feats}")

    def predict(self, spark_df):
        # 广播模型和特征信息
        model_broadcast = spark_df._sc.broadcast(self.model)
        cat_info_broadcast = spark_df._sc.broadcast(self.cat_info)
        cat_features_broadcast = spark_df._sc.broadcast(self.cat_features)
        num_features_broadcast = spark_df._sc.broadcast(self.num_features)
        feature_names = self.feature_names

        def predict_batch(pdf):
            """处理单批数据并返回DataFrame"""
            model = model_broadcast.value
            cat_info = cat_info_broadcast.value
            cat_features = cat_features_broadcast.value
            num_features = num_features_broadcast.value

            if len(pdf) == 0:
                return pd.DataFrame(columns=['id', 'prediction', 'probability'])

            # 提取ID
            ids = pdf['id'].tolist() if 'id' in pdf.columns else [f"unknown_{i}" for i in range(len(pdf))]

            # 处理特征
            features = {f: [] for f in feature_names if f in pdf.columns}

            # 处理分类特征
            for feat in cat_features:
                if feat not in pdf.columns:
                    continue
                # 创建类别到编码的映射
                cat_map = {str(cat): idx for idx, cat in enumerate(cat_info[feat]['original'])}
                cat_map['unknown'] = -1  # 未知类别映射为-1
                for val in pdf[feat].fillna('unknown').astype(str).tolist():
                    features[feat].append(cat_map.get(val, -1))

            # 处理数值特征
            for feat in num_features:
                if feat not in pdf.columns:
                    continue
                for val in pdf[feat].tolist():
                    if pd.isna(val):
                        features[feat].append(0.0)
                    else:
                        features[feat].append(float(val))

            # 创建特征DataFrame
            X = pd.DataFrame(features)
            # 确保特征顺序与模型训练时一致
            model_feats = model.feature_name()
            X = X.reindex(columns=model_feats, fill_value=0)

            # 执行预测
            probabilities = model.predict(X).tolist()
            predictions = [int(p > 0.5) for p in probabilities]

            # 构造结果
            return pd.DataFrame({
                'id': ids,
                'prediction': predictions,
                'probability': probabilities
            })

        # 使用分区ID进行分组预测
        spark_df = spark_df.withColumn("group_key", spark_partition_id())
        result_df = spark_df.groupby("group_key").applyInPandas(
            predict_batch,
            schema=self.output_schema
        )

        return result_df.drop("group_key")


# 主函数
def main():
    model_path = "lightgbm_model.txt"
    meta_path = "feature_meta_data.pkl"
    data_path = "training_data.csv"  # 由generate_training_data.py生成的文件
    spark = init_spark()

    try:
        print("开始训练模型...")
        # 训练模型并保存特征信息
        cat_features, num_features, cat_info = train_lightgbm_model(spark, model_path, meta_path, data_path)

        # 模拟预测阶段：加载特征信息
        print("加载特征元数据...")
        loaded_cat_feats, loaded_num_feats, loaded_all_feats, loaded_cat_info = load_meta_data(meta_path)

        print("生成测试数据...")
        # 使用加载的特征信息初始化预测器
        predictor = LightGBMPredictor(
            model_path,
            loaded_cat_feats,
            loaded_num_feats,
            loaded_all_feats,
            loaded_cat_info
        )

        # 加载测试数据
        test_df = load_training_data(spark, "training_data.csv")
        # 为测试数据添加id列用于结果标识
        test_df = test_df.withColumn("id", col("user_id"))
        test_df = test_df.limit(10)

        print("预处理测试数据...")
        processor = FeatureProcessor(loaded_cat_info, loaded_cat_feats, loaded_num_feats)
        processed_test_df = processor.process(test_df)

        print("执行预测...")
        result_df = predictor.predict(processed_test_df)

        print("预测结果示例:")
        result_df.show(10)

        result_df.write.mode("overwrite").csv("prediction_results", header=True)
        print("预测结果已保存至: prediction_results")

    except Exception as e:
        print(f"执行出错: {str(e)}")
    finally:
        spark.stop()
        print("Spark会话已关闭")

if __name__ == "__main__":
    main()