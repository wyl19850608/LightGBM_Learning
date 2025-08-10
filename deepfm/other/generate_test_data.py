import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# 设置随机种子，保证结果可复现
np.random.seed(42)
random.seed(42)

# 特征列表
numeric_features = [
    'age', 'last_30d_tel_succ_cs', 'loss_model_ggroup_v3',
    'risk_ms11_1_model_score', 'standard_score_group_v6_1',
    'last_month_expire_coupon_cnt', 'number_of_gold_be_used',
    'last_10d_lin_e_cnt', 'last_10d_gu_e_cnt',
    'ayht10_all_respond_score', 'call_anss_score_t10',
    'xyl_model_107', 'avail_cash', 'avg_limuse_rate',
    'pril_bal', 'crdt_lim_yx', 'lim_use_rate', 'zaidai_ctrl_rate'
]

categorical_features = [
    'yls_cust_type_v2', 'cust_types_01', 'cust_types_02',
    'if_sms_yn', 'elec_types', 'igender_cd', 'icust_ty',
    'if_500up_yn', 'is_login', 'sms_types', 'if_bj_30_yn',
    'member_types', 'if_bj_10_yn'
]

# 明确指定日期时间特征
datetime_features = [
    'invite_time', 'last_fq_dt', 'last_tx_dt', 'last_op_time',
    'last_pboc_cx_dtуспсq03nwww.', 'last_login_ayh_time',
    'last_login_app_time', 'last_uas_reject_time',
    'last_yunying_jj_beegin_time', 'last_yunying_jj_end_time'
]

# 其他特征 - 输出顺序将按此列表排列
other_features = [
    'user_id', 'unique_id',
    'gender_cd', 'residence_city_cd', 'house_hold_type',
    'marital_stattus_cd', 'edu_deg_cd', 'm_income', 'income_ind',
    'cust_ty', 'custtypes_01', 'yls_cust_type',
    'xyl_tag', 'last_fq_dt', 'last_fq_txk_amt', 'last_tx_dt',
    'last_tx_amt', 'last_30d_login_app_days', 'last_30d_push_touch_times',
    'last_30d_sms_send_succ_cs', 'last_30d_tel_sCC_cs', 'last_5d_coupon_cnt',
    'last_10d_coupon_cnt', 'last_op_time', 'last_pboc_cx_dtуспсq03nwww.',
    'querycncq03wwlw', 'querycncebookswwflaged', 'сdaccapwwwww.',
    'гvaccapwwwwww.', 'гуграфиям', 'густармим', 'rvclaapwwww.',
    'rvblsapwwwww.', 'rvnbmwww.', 'rvnbawpwwwww.', 'rvapsappywww.',
    'rvapmapwwwwww.', 'rvapaapwwwww.', 'repay_date_days',
    'als_m1_id_bank_allnum', 'als_m1_id_nbank_allnum',
    'als_m3_id_bank_allnum', 'als_m3_id_nbank_allnum', 'br_modle_score',
    'query_times_bank_90d', 'query_imes_cfc_90d', 'risk_ms9_model_score',
    'loss_model_group_v3', 'yls_cust_typev2', 'latest_login_days',
    'total_loan_cnt', 'total_loan_amt', 'last_1y_cps_num',
    'cuir_mon_use_gold_cnt', 'late_1mon_use_gold_cnt',
    'late_3mon_use_gold_इंडो', 'late_6mon_use_gold_cnt', 'cur_mon_use_points_cnt',
    'late_1mon_use_points_cnt', 'late_3mon_use_points_cnt', 'late_6mon_use_points_cnt',
    'cur_mon_poins_sign_cnt', 'late_imon_points_sign_cnt', 'late_3mon_points_sign_cnt',
    'late_6mon_points_sign_cnt', 'cur_mon_points_luck_gift_cnt',
    'late_1Imon_points_luck_gift_cnt', 'late_3mon_points_luck_gift_cnt',
    'late_6mon_points_luck_gift_cnt', 'cur_mon_cps_order_types',
    'late_imon_cps_order_types', 'late_3mon_cps_order_types',
    'late_6mon_cps_order_types', 'last_mon_game_accum_click_cnt',
    'last_mon_game_accum_expo_cnt', 'last_mon_read_accum_click_cnt',
    'last_mon_read_accum_expo_cnt', 'cur_yili_vipterms', 'cur_juhui_vip_terms',
    'last_login_ayh_time', 'last_login_app_time', 'last_uas_reject_time',
    'last_30d_fq_cs', 'last_30d_tx_cs', 'last_30d_login_ayh_days',
    'last_5d_lin_e_cnt', 'last_Sd_gu_e_cnt',
    'last_yunying_jj_new_rate', 'last_yunying_jj_beegin_time',
    'last_yunying_jj_end_time', 'last_90d_yunying_jj_cs', 'last_180d_yunying_jj_cs',
    'partner_yx_contr_cnt', 'ayht10_all_respond_score', 'call_ans_score_t10',
    'member_types', 'xyl_model_107', 'kzr_types', 'elec_types', 'sms_types',
    'if_sms_yn', 'if_ele_yn', 'if_500up_yn', 'if_bj_10_yn', 'if_bj_30_yn',
    'avail_bal_cash', 'send_coupon_cnt_1mon', 'use_coupon_cnt_1mon',
    'valid_coupon_cnt', 'coupon_use', 'clkk_wyyh_count', 'clk_llydg_count',
    'clk_yhq_sy_count', 'clk_hyzx_sy_count', 'clk_jryhq_count', 'zaidai_days',
    'zaidai_days/mob_3y as zaidai_rate', 'avg_lim_use_rate', 'pril_bal',
    'crdt_lim_yx', 'lim_use_rate', 'zaidai_ctrl_rate', 'is_login',
    'list_call_times_d', 'list_ansr_times_d', 'list_ansr_dur_d',
    'list_manu_call_times_d', 'list_manu_ansr_times_d', 'list_manu_ansr_dur_d',
    'list_ivr_call_times_d', 'list_ivr_ansr_timees_d', 'list_ivr_ansr_dur_d',
    'list_call_times', 'list_ansr_times', 'list_ansr_dur', 'cust_call_times',
    'cust_ansr_times', 'cust_ansr_dur', 'cust_call_times_mon',
    'cust_ansr_times_mon', 'cust_ansr_dur_mon', 'cust_manu_call_times',
    'cust_manu_ansr_times', 'cust_manu_ansr_dur', 'cust_manu_call_times_mon',
    'cust_manu_ansr_times_mon', 'cust_manu_ansr_dur_mon', 'cust_ivr_call_times',
    'cust_ivr_ansr_times', 'cust_ivr_ansr_dur', 'otust_ivr_call_times_mon',
    'cust_ivr_ansr_times_mon', 'cust_ivr_ansr_dur_mon'
]

# 生成1000行数据
n_rows = 10000
data = {}

# 生成数值特征（连续值）
data['age'] = np.random.randint(18, 70, n_rows)  # 年龄：18-70岁
data['last_30d_tel_succ_cs'] = np.random.randint(0, 20, n_rows)  # 30天电话成功次数
data['loss_model_ggroup_v3'] = np.random.normal(0.5, 0.1, n_rows).clip(0, 1)  # 损失模型分数：0-1
data['risk_ms11_1_model_score'] = np.random.normal(500, 100, n_rows).clip(300, 850)  # 风险分数：300-850
data['standard_score_group_v6_1'] = np.random.randint(1, 10, n_rows)  # 标准分组成员：1-9
data['last_month_expire_coupon_cnt'] = np.random.randint(0, 15, n_rows)  # 上月过期优惠券数
data['number_of_gold_be_used'] = np.random.randint(0, 50, n_rows)  # 金币使用数量
data['last_10d_lin_e_cnt'] = np.random.randint(0, 30, n_rows)  # 10天内林e次数
data['last_10d_gu_e_cnt'] = np.random.randint(0, 25, n_rows)  # 10天内谷e次数
data['ayht10_all_respond_score'] = np.random.normal(70, 15, n_rows).clip(0, 100)  # 响应分数：0-100
data['call_anss_score_t10'] = np.random.normal(60, 20, n_rows).clip(0, 100)  # 呼叫应答分数
data['xyl_model_107'] = np.random.normal(0.6, 0.2, n_rows).clip(0, 1)  # 模型分数：0-1
data['avail_cash'] = np.random.lognormal(10, 1, n_rows).clip(1000, 1000000)  # 可用现金：对数分布
data['avg_limuse_rate'] = np.random.uniform(0, 1, n_rows)  # 平均限额使用率：0-1
data['pril_bal'] = np.random.lognormal(8, 1.5, n_rows).clip(100, 100000)  # 上月余额
data['crdt_lim_yx'] = np.random.lognormal(9, 1, n_rows).clip(1000, 500000)  # 信用额度
data['lim_use_rate'] = np.random.uniform(0, 1.2, n_rows).clip(0, 1.5)  # 限额使用率
data['zaidai_ctrl_rate'] = np.random.uniform(0, 1, n_rows)  # 再贷控制率

# 生成类别特征
data['yls_cust_type_v2'] = np.random.choice(['A', 'B', 'C', 'D', 'E'], n_rows)  # 客户类型
data['cust_types_01'] = np.random.choice(['VIP', '普通', '新客户', '流失风险'], n_rows)  # 客户类型1
data['cust_types_02'] = np.random.choice([1, 2, 3, 4], n_rows)  # 客户类型2
data['if_sms_yn'] = np.random.choice(['Y', 'N'], n_rows, p=[0.7, 0.3])  # 是否接收短信
data['elec_types'] = np.random.choice(['类型1', '类型2', '类型3', '未知'], n_rows)  # 电子类型
data['igender_cd'] = np.random.choice([1, 2, 9], n_rows, p=[0.5, 0.49, 0.01])  # 性别编码
data['icust_ty'] = np.random.choice([10, 20, 30, 40, 50], n_rows)  # 客户类型编码
data['if_500up_yn'] = np.random.choice(['Y', 'N'], n_rows, p=[0.3, 0.7])  # 是否500以上
data['is_login'] = np.random.choice([0, 1], n_rows, p=[0.2, 0.8])  # 是否登录
data['sms_types'] = np.random.choice(['营销', '通知', '验证', '其他'], n_rows)  # 短信类型
data['if_bj_30_yn'] = np.random.choice(['Y', 'N'], n_rows, p=[0.1, 0.9])  # 是否30天标记
data['member_types'] = np.random.choice(['普通会员', '白银', '黄金', '铂金', '钻石'], n_rows)  # 会员类型
data['if_bj_10_yn'] = np.random.choice(['Y', 'N'], n_rows, p=[0.15, 0.85])  # 是否10天标记

# 生成标签（0或1）- 稍微不平衡
data['label'] = np.random.choice([0, 1], n_rows, p=[0.8, 0.2])

# 生成其他特征
data['user_id'] = [f'user_{i:04d}' for i in range(n_rows)]  # 用户ID
data['unique_id'] = [f'unique_{random.getrandbits(32):08x}' for _ in range(n_rows)]  # 唯一ID

# 生成日期特征
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)
date_range = (end_date - start_date).days

# 为每个日期特征生成随机日期
for feature in datetime_features:
    data[feature] = [start_date + timedelta(days=random.randint(0, date_range))
                     for _ in range(n_rows)]

# 随机生成其他特征
for feature in other_features:
    if feature in data:  # 已生成的特征跳过
        continue

    # 随机决定是类别型还是数值型
    if random.random() < 0.3:  # 30%概率生成类别特征
        n_categories = random.randint(2, 5)
        categories = [f'类别{i+1}' for i in range(n_categories)]
        data[feature] = np.random.choice(categories, n_rows)
    else:  # 70%概率生成数值特征
        if 'count' in feature or 'cnt' in feature or 'times' in feature:  # 计数类特征
            data[feature] = np.random.randint(0, 100, n_rows)
        elif 'score' in feature:  # 分数类特征
            data[feature] = np.random.normal(50, 20, n_rows).clip(0, 100)
        elif 'days' in feature:  # 天数类特征
            data[feature] = np.random.randint(0, 365, n_rows)
        elif 'amt' in feature or 'cash' in feature or 'bal' in feature:  # 金额类特征
            data[feature] = np.random.lognormal(6, 2, n_rows).clip(10, 100000)
        else:  # 其他数值特征
            data[feature] = np.random.normal(0, 1, n_rows)

# 转换为DataFrame并按指定顺序排列列
# 列顺序: other_features + numeric_features + categorical_features + [label]
all_columns = (other_features +
               numeric_features +
               categorical_features +
               ['label'])

# 确保没有重复列
all_columns = list(dict.fromkeys(all_columns))
print("*"*30)
print(all_columns)

# 创建DataFrame并按指定顺序排列
df = pd.DataFrame(data)[all_columns]

# 只对明确的日期特征进行格式转换，避免错误
for col in datetime_features:
    if col in df.columns:
        # 明确指定格式，避免解析警告
        df[col] = pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S').dt.strftime('%Y-%m-%d %H:%M:%S')

# 保存为CSV文件
df.to_csv('test_data_ordered.csv', index=False)
print(f"已生成10000行测试数据，保存至 test_data_ordered.csv")
print(f"数据形状: {df.shape}")
print(f"标签分布: {dict(df['label'].value_counts())}")
print(f"列顺序已按 other_features + 数值特征 + 类别特征 + label 排列")
