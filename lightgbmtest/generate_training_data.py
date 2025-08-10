import pandas as pd
import numpy as np
import random
from faker import Faker

# 初始化Faker用于生成更真实的模拟数据
fake = Faker('zh_CN')

# 定义所有特征列
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

# 定义分类特征
categorical_features = [
    'is_coupon_used', 'push_status', 'delay_type', 'tel_call_type',
    'touch_status', 'click_status', "is_coupon_issue", 'is_credit',
    'is_apply', "is_wdraw", "aprv_status", "is_realname_cert",
    'is_mobile_realname_cert', 't19_is_complete_base_info',
    't19_is_complete_occ_info', 't19_is_complete_contact_info',
    'reg_dev_compile_id', 'is_ms_staff', 'gender_cd',
    "residence_city_cd", "ocr_ethnic", "residence_pr_cd",
    'occ_cd', "census_pr_cd", 'census_city_cd', "census_area_cd",
    'residence_area_cd', 'cust_ty', 'cust_types_01', 'cust_types_02',
    'yls_cust_type', 'flagcd', 'kzr_types', 'elec_types', 'sms_types',
    'if_sms_yn', 'if_ele_yn', 'if_500up_yn', 'if_bj_10_yn',
    'if_bj_30_yn', 'is_login', 'db_src', 't19_user_status', 'user_mobile_status',
    'reg_dev_app_dl_chan_no', 'reg_chan_no', "reg_dev_info",
    'reg_dev_sys_type', 'reg_dev_sys_vn', 'reg_app_vn',
    'reg_dev_mfr', 'reg_dev_brand', 'reg_dev_model', 'reg_imei'
]

# 连续特征是总特征减去分类特征
continuous_features = [col for col in COLUMNS_DF if col not in categorical_features]

def generate_categorical_data(feature_name, n_samples):
    """生成分类特征数据"""
    # 为不同的分类特征定义可能的取值范围
    if feature_name in ['is_coupon_used', 'is_credit', 'is_apply', 'is_wdraw',
                        'is_realname_cert', 'is_mobile_realname_cert', 't19_is_complete_base_info',
                        't19_is_complete_occ_info', 't19_is_complete_contact_info', 'is_ms_staff',
                        'if_sms_yn', 'if_ele_yn', 'if_500up_yn', 'if_bj_10_yn', 'if_bj_30_yn', 'is_login']:
        # 布尔型特征，0或1
        return np.random.choice([0, 1], size=n_samples)

    elif feature_name in ['push_status', 'delay_type', 'tel_call_type', 'touch_status',
                          'click_status', 'is_coupon_issue', 'aprv_status', 't19_user_status',
                          'user_mobile_status', 'kzr_types', 'elec_types', 'sms_types', 'flagcd']:
        # 多状态分类特征
        return np.random.choice(range(5), size=n_samples)  # 0-4的状态值

    elif feature_name in ['db_src', 'reg_dev_app_dl_chan_no', 'reg_chan_no', 'reg_attr_chan_no',
                          'chan_type', 'reg_dev_sys_type', 'reg_dev_mfr', 'reg_dev_brand',
                          'cust_ty', 'cust_types_01', 'cust_types_02', 'yls_cust_type']:
        # 渠道或类型编码
        return [f"code_{random.randint(100, 999)}" for _ in range(n_samples)]

    elif feature_name in ['reg_dev_info', 'reg_dev_model', 'reg_dev_sys_vn', 'reg_app_vn']:
        # 设备信息相关
        return [f"info_{random.randint(10, 99)}.{random.randint(1, 9)}" for _ in range(n_samples)]

    elif feature_name in ['residence_city_cd', 'residence_pr_cd', 'census_pr_cd',
                          'census_city_cd', 'census_area_cd', 'residence_area_cd', 'occ_cd']:
        # 地区或职业编码
        return [f"{random.randint(100000, 999999)}" for _ in range(n_samples)]

    elif feature_name == 'ocr_ethnic':
        # 民族
        ethnicities = ['汉', '蒙古', '回', '藏', '维吾尔', '苗', '彝', '壮', '布依', '朝鲜']
        return np.random.choice(ethnicities, size=n_samples)

    elif feature_name == 'gender_cd':
        # 性别
        return np.random.choice(['男', '女', '未知'], size=n_samples)

    elif feature_name in ['reg_dev_compile_id', 'reg_imei']:
        # 设备唯一标识
        return [fake.uuid4()[:16] for _ in range(n_samples)]

    else:
        # 默认返回随机整数分类
        return np.random.choice(range(3), size=n_samples)

def generate_continuous_data(feature_name, n_samples):
    """生成连续特征数据"""
    if feature_name in ['user_id', 'unique_id', 'subscribe_no', 'prod_cd']:
        # ID类特征
        return [random.randint(10000000, 99999999) for _ in range(n_samples)]

    elif feature_name in ['delay_days', 'push_cnt', 'sms_charge_cnt', 'tel_call_dur',
                          'wdraw_cnt', 'tel_call_inav_cnt', 'apply_cnt', 'age',
                          'last_30d_login_app_days', 'last_30d_push_touch_times',
                          'last_30d_sms_send_succ_cs', 'last_30d_tel_succ_cs',
                          'last_5d_coupon_cnt', 'last_10d_coupon_cnt', 'pboc_repay_days',
                          'als_m1_id_bank_allnum', 'als_mi_id_nbank_allnum',
                          'als_m3_id_bank_allnum', 'als_m3_id_nbank_allnum',
                          'query_times_bank_90d', 'query_times_cfc_90d',
                          'last_30d_fq_cs', 'last_30d_tx_cs', 'last_30d_login_ayh_days',
                          'last_5d_lin_e_cnt', 'last_10d_lin_e_ctnt',
                          'last_5d_gu_e_cnt', 'last_10d_gu_e_cnt',
                          'last_90d_yunying_jj_cs', 'last_180d_yunying_jj_cs',
                          'partner_yx_contr_cnt', 'latest_login_days',
                          'total_loan_cnt']:
        # 计数或天数类特征，非负整数
        return np.random.randint(0, 100, size=n_samples)

    elif feature_name in ['wdraw_amt', 'wdraw_amt_t0', 'wdraw_amt_t3', 'wdraw_amt_t7',
                          'wdraw_amt_t10', 'wdraw_amt_t14', 'wdraw_amt_t15', 'wdraw_amt_t30',
                          'apply_amt', 'last_fq_tx_amt', 'last_tx_amt',
                          'querycncq03nwww', 'querycncq03wwlw', 'querycncq03wwpw',
                          'cdaccapwwwwww', 'diaccapwwwwwww', 'rvaccapwwww',
                          'rvrpsapwwwww', 'rvclmapwww', 'rvclaapwwww',
                          'rewblsapwwww', 'rvnbmwpwwwww', 'rvnbawpwwwww',
                          'rvapsapwwwwww', 'rvapmapwwww', 'rvapaapwwwww',
                          'total_loan_amt', 'avail_bal_cash', 'pril_bal',
                          'crdt_lim_yx']:
        # 金额类特征，可能是较大的数值
        return np.random.uniform(0, 100000, size=n_samples).round(2)

    elif feature_name in ['last_yunying_jj_new_rate', 'avg_lim_use_rate', 'lim_use_rate']:
        # 比率类特征，通常在0-1或0-100之间
        return np.random.uniform(0, 100, size=n_samples).round(4)

    elif feature_name in ['last_yunying_jj_begin_time', 'last_yunying_jj_end_time']:
        # 时间类特征
        return [fake.date_between(start_date='-1y', end_date='today') for _ in range(n_samples)]

    elif feature_name == 'phy_del_op_side':
        # 物理删除操作方，这里作为连续特征可能代表操作次数
        return np.random.randint(0, 5, size=n_samples)

    else:
        # 默认返回0-100的随机数
        return np.random.uniform(0, 100, size=n_samples).round(2)

def generate_training_data(n_samples=1000):
    """生成完整的训练数据"""
    data = {}

    # 生成分类特征数据
    for feature in categorical_features:
        data[feature] = generate_categorical_data(feature, n_samples)

    # 生成连续特征数据
    for feature in continuous_features:
        data[feature] = generate_continuous_data(feature, n_samples)

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 确保列的顺序与COLUMNS_DF一致
    df = df[COLUMNS_DF]

    return df

if __name__ == "__main__":
    # 生成10000条训练数据
    n_samples = 10000
    print(f"开始生成{ n_samples }条训练数据...")

    training_data = generate_training_data(n_samples)

    # 保存为CSV文件
    output_file = "training_data.csv"
    training_data.to_csv(output_file, index=False, encoding='utf-8')

    print(f"训练数据生成完成，共{ len(training_data) }行，{ len(training_data.columns) }列")
    print(f"数据已保存至: { output_file }")

    # 显示前5行数据预览
    print("\n数据预览:")
    print(training_data.head())

    # 显示特征类型统计
    print("\n特征类型统计:")
    print(f"分类特征数量: { len(categorical_features) }")
    print(f"连续特征数量: { len(continuous_features) }")
    print(f"总特征数量: { len(COLUMNS_DF) }")
