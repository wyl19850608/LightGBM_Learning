import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta

# 初始化Faker
fake = Faker('zh_CN')

# 设置随机种子，保证结果可复现
np.random.seed(42)
random.seed(42)

def generate_credit_data(n_samples=1000, output_file='full_credit_data.csv'):
    """
    生成完整的信贷测试数据，包含所有字段和随机空值

    参数:
        n_samples: 生成的数据行数
        output_file: 输出文件路径
    """
    # 生成目标变量 (0或1) - 0表示拒绝，1表示通过
    labels = np.random.randint(0, 2, size=n_samples)

    # 空值概率配置（根据字段重要性设置不同空值率）
    null_prob = {
        # 身份与地域信息类
        'age': 0.02,
        'gender_cd': 0.03,
        'ocr_ethnic': 0.1,
        'residence_pr_cd': 0.05,
        'residence_city_cd': 0.08,
        'residence_area_cd': 0.1,
        'census_pr_cd': 0.08,
        'census_city_cd': 0.1,
        'census_area_cd': 0.15,
        'occ_cd': 0.1,

        # 信贷核心信息类
        'aprv_status': 0.0,  # 目标变量无空值
        'pril_bal': 0.01,
        'crdt_lim_yx': 0.01,
        'avail_bal_cash': 0.01,
        'total_loan_cnt': 0.02,
        'total_loan_amt': 0.02,
        'apply_cnt': 0.02,
        'apply_amt': 0.02,
        'wdraw_cnt': 0.02,
        'wdraw_amt': 0.02,
        'lim_use_rate': 0.01,

        # 时间与触达信息类
        'cust_recv_time': 0.03,
        'touch_time': 0.3,
        'cust_date': 0.03,
        'push_cnt': 0.05,
        'sms_charge_cnt': 0.08,
        'tel_call_type': 0.2,
        'tel_call_dur': 0.2,
        'tel_call_inav_cnt': 0.2,
        'touch_status': 0.1,
        'click_status': 0.2,

        # 行为交互信息类
        'is_coupon_issue': 0.05,
        'is_credit': 0.05,
        'is_apply': 0.05,
        'is_wdraw': 0.05,
        'is_login': 0.05,
        'delay_days': 0.1,
        'if_bj_10_yn': 0.1,
        'if_bj_30_yn': 0.1,
        'wdraw_amt_t0': 0.2,
        'wdraw_amt_t3': 0.2,
        'wdraw_amt_t7': 0.2,
        'wdraw_amt_t10': 0.2,
        'wdraw_amt_t14': 0.2,
        'wdraw_amt_t15': 0.2,
        'wdraw_amt_t30': 0.02,

        # 业务系统信息类
        'first_biz_name': 0.03,
        'second_biz_name': 0.05,
        'third_biz_name': 0.08,
        'db_src': 0.03,
        'prod_cd': 0.03,
        'cust_gp_code': 0.05,
        'cust_gp_name': 0.05,
        'touch_name': 0.1,
        'touch_id': 0.1,
        'touch_type': 0.1,
        'channel_name': 0.05,
        'channel_id': 0.05,
        'plan_id': 0.08,
        'plan_name': 0.08,
        'channel_task_id': 0.1,
        'subscribe_no': 0.15,
        'decision_id': 0.1,
        'decision_name': 0.1,
        't1_cust_id': 0.2,
        't2_cust_id': 0.3
    }

    # 工具函数：为特征添加空值
    def add_nulls(arr, prob, is_integer=False):
        """
        根据概率为数组添加空值

        参数:
            arr: 原始数组
            prob: 空值概率
            is_integer: 是否为整数类型（需要特殊处理，因为整数不支持NaN）
        """
        if len(arr) == 0:
            return arr

        # 创建掩码，随机选择要设为空值的元素
        mask = np.random.random(len(arr)) < prob
        print("mask", mask)
        print("mask.type", type(mask))

        # 处理整数类型（先转换为float类型以支持NaN）
        if is_integer:
            result = arr.astype(float)
            result[mask] = np.nan
            return result

        # 处理不同类型的数组
        if isinstance(arr, np.ndarray):
            # 对于numpy数组
            result = arr.copy()
            # 确保数组可以容纳NaN
            if np.issubdtype(result.dtype, np.integer):
                result = result.astype(float)
            result[mask] = np.nan
        else:
            # 对于列表
            result = [np.nan if mask[i] else val for i, val in enumerate(arr)]

        return result

    # 1. 生成身份与地域信息类特征
    # 年龄：18-65岁（整数，需要特殊处理空值）
    age = add_nulls(np.random.randint(18, 65, size=n_samples), null_prob['age'], is_integer=True)

    # 性别代码：1-男，2-女，9-未知
    gender_cd = add_nulls(
        np.random.choice(['1', '2', '9'], size=n_samples, p=[0.5, 0.49, 0.01]),
        null_prob['gender_cd']
    )

    # 民族
    ocr_ethnic = add_nulls(
        np.random.choice(['汉族', '蒙古族', '回族', '藏族', '维吾尔族', '壮族', '满族', '其他'],
                         size=n_samples, p=[0.9, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.02]),
        null_prob['ocr_ethnic']
    )

    # 地区代码（简化版）
    province_codes = [f'{i:02d}' for i in range(11, 82)]  # 模拟实际省份代码范围
    city_codes = [f'{i:04d}' for i in range(1101, 8101, 100)]  # 模拟城市代码
    area_codes = [f'{i:06d}' for i in range(110101, 810101, 1000)]  # 模拟区县代码

    residence_pr_cd = add_nulls(np.random.choice(province_codes, size=n_samples), null_prob['residence_pr_cd'])
    residence_city_cd = add_nulls(np.random.choice(city_codes, size=n_samples), null_prob['residence_city_cd'])
    residence_area_cd = add_nulls(np.random.choice(area_codes, size=n_samples), null_prob['residence_area_cd'])
    census_pr_cd = add_nulls(np.random.choice(province_codes, size=n_samples), null_prob['census_pr_cd'])
    census_city_cd = add_nulls(np.random.choice(city_codes, size=n_samples), null_prob['census_city_cd'])
    census_area_cd = add_nulls(np.random.choice(area_codes, size=n_samples), null_prob['census_area_cd'])

    # 职业代码
    occ_codes = [f'{i:03d}' for i in range(100, 900, 50)]  # 模拟职业代码
    occ_cd = add_nulls(np.random.choice(occ_codes, size=n_samples), null_prob['occ_cd'])

    # 2. 生成信贷核心信息类特征
    # 在贷余额（不含会员、溢缴款）
    pril_bal = add_nulls(
        np.random.lognormal(8, 1.5, size=n_samples).round(2),  # 使用对数正态分布模拟金额
        null_prob['pril_bal']
    )

    # 当前有效额度
    crdt_lim_yx = add_nulls(
        np.random.lognormal(9, 1.2, size=n_samples).round(2),
        null_prob['crdt_lim_yx']
    )

    # 现金分期可用额度
    avail_bal_cash = add_nulls(
        np.maximum(0, np.where(~np.isnan(crdt_lim_yx) & ~np.isnan(pril_bal),
                               crdt_lim_yx - pril_bal, 0)).round(2),
        null_prob['avail_bal_cash']
    )

    # 累计放款次数（整数）
    total_loan_cnt = add_nulls(
        np.random.randint(0, 20, size=n_samples),
        null_prob['total_loan_cnt'],
        is_integer=True
    )

    # 累计放款金额
    total_loan_amt = add_nulls(
        np.where(total_loan_cnt > 0,
                 np.random.lognormal(10, 1.5, size=n_samples).round(2),
                 0),
        null_prob['total_loan_amt']
    )

    # 有效期内提现申请总次数（整数）
    apply_cnt = add_nulls(
        np.random.randint(0, 10, size=n_samples),
        null_prob['apply_cnt'],
        is_integer=True
    )

    # 有效期内提现申请总金额
    apply_amt = add_nulls(
        np.where(apply_cnt > 0,
                 np.random.lognormal(9, 1.2, size=n_samples).round(2),
                 0),
        null_prob['apply_amt']
    )

    # 有效期内提现成功总次数（整数）
    wdraw_cnt = add_nulls(
        np.minimum(
            np.where(np.isnan(apply_cnt), 0, apply_cnt),
            np.random.randint(0, 10, size=n_samples)
        ),
        null_prob['wdraw_cnt'],
        is_integer=True
    )

    # 有效期内提现成功总金额
    wdraw_amt = add_nulls(
        np.where(wdraw_cnt > 0,
                 np.random.uniform(0.5, 1.0, size=n_samples) *
                 np.where(np.isnan(apply_amt), 0, apply_amt),
                 0).round(2),
        null_prob['wdraw_amt']
    )

    # 额度使用率
    lim_use_rate = add_nulls(
        np.where(
            np.logical_and(~np.isnan(pril_bal), pril_bal > 0) &
            np.logical_and(~np.isnan(crdt_lim_yx), crdt_lim_yx > 0),
            (pril_bal / crdt_lim_yx).round(4),
            0
        ),
        null_prob['lim_use_rate']
    )

    # 3. 生成时间与触达信息类特征
    # 生成基础日期
    base_dates = [fake.date_between(start_date='-1y', end_date='today') for _ in range(n_samples)]

    # 名单接收日期
    cust_recv_time = add_nulls(
        [d.strftime('%Y-%m-%d') for d in base_dates],
        null_prob['cust_recv_time']
    )

    # 首次触达时间
    touch_time = add_nulls(
        [fake.date_time_between(start_date=d, end_date='+30d').strftime('%Y-%m-%d %H:%M:%S')
         if random.random() > 0.3 else None for d in base_dates],
        null_prob['touch_time']
    )

    # 名单日期
    cust_date = add_nulls(
        [d.strftime('%Y-%m-%d') for d in base_dates],
        null_prob['cust_date']
    )

    # push推送终端数（整数）
    push_cnt = add_nulls(
        np.random.randint(0, 10, size=n_samples),
        null_prob['push_cnt'],
        is_integer=True
    )

    # 短信计费条数（整数）
    sms_charge_cnt = add_nulls(
        np.random.randint(0, 5, size=n_samples),
        null_prob['sms_charge_cnt'],
        is_integer=True
    )

    # 电销拨打类型
    tel_call_type = add_nulls(
        np.random.choice(['0', '1', '2', '3', ''], size=n_samples, p=[0.2, 0.3, 0.2, 0.1, 0.2]),
        null_prob['tel_call_type']
    )

    # 电销最大拨打时长（秒）（整数）
    # 修复逻辑运算错误：添加括号并使用逻辑与运算符
    tel_call_dur = add_nulls(
        np.where((tel_call_type != '') & (~pd.isna(tel_call_type)),
                 np.random.randint(10, 300, size=n_samples),
                 0),
        null_prob['tel_call_dur'],
        is_integer=True
    )

    # 电销最大交互轮次（整数）
    # 修复逻辑运算错误：添加括号并使用逻辑与运算符
    tel_call_inav_cnt = add_nulls(
        np.where((tel_call_type != '') & (~pd.isna(tel_call_type)),
                 np.random.randint(1, 10, size=n_samples),
                 0),
        null_prob['tel_call_inav_cnt'],
        is_integer=True
    )

    # 触达状态
    touch_status = add_nulls(
        np.random.choice(['成功', '失败', '未触达'], size=n_samples, p=[0.6, 0.2, 0.2]),
        null_prob['touch_status']
    )

    # 点击状态
    click_status = add_nulls(
        np.random.choice(['点击', '未点击', ''], size=n_samples, p=[0.3, 0.5, 0.2]),
        null_prob['click_status']
    )

    # 4. 生成行为交互信息类特征
    # 是否发放优惠券
    is_coupon_issue = add_nulls(
        np.random.choice(['Y', 'N'], size=n_samples, p=[0.3, 0.7]),
        null_prob['is_coupon_issue']
    )

    # 是否发起授信申请
    is_credit = add_nulls(
        np.random.choice(['Y', 'N'], size=n_samples, p=[0.2, 0.8]),
        null_prob['is_credit']
    )

    # 是否发起提现申请
    is_apply = add_nulls(
        np.random.choice(['Y', 'N'], size=n_samples, p=[0.15, 0.85]),
        null_prob['is_apply']
    )

    # 是否提现成功
    # 修复逻辑运算错误：添加括号并使用逻辑与运算符
    is_wdraw = add_nulls(
        np.where((is_apply == 'Y') & (~pd.isna(is_apply)),
                 np.random.choice(['Y', 'N'], size=n_samples, p=[0.5, 0.5]),
                 'N'),
        null_prob['is_wdraw']
    )

    # 是否登录
    is_login = add_nulls(
        np.random.choice(['Y', 'N'], size=n_samples, p=[0.6, 0.4]),
        null_prob['is_login']
    )

    # 延迟天数（整数）
    delay_days = add_nulls(
        np.random.randint(0, 15, size=n_samples),
        null_prob['delay_days'],
        is_integer=True
    )

    # 近10天内是否提现被拒
    if_bj_10_yn = add_nulls(
        np.random.choice(['Y', 'N'], size=n_samples, p=[0.1, 0.9]),
        null_prob['if_bj_10_yn']
    )

    # 近30天内是否提现被拒
    if_bj_30_yn = add_nulls(
        np.random.choice(['Y', 'N'], size=n_samples, p=[0.15, 0.85]),
        null_prob['if_bj_30_yn']
    )

    # 各时间段提现金额
    # 修复逻辑运算错误：添加括号并使用逻辑与运算符
    wdraw_amt_t0 = add_nulls(
        np.where((is_wdraw == 'Y') & (~pd.isna(is_wdraw)),
                 np.random.uniform(0, np.where(np.isnan(wdraw_amt), 0, wdraw_amt), size=n_samples).round(2),
                 0),
        null_prob['wdraw_amt_t0']
    )

    # 修复逻辑运算错误：添加括号并使用逻辑与运算符
    wdraw_amt_t3 = add_nulls(
        np.where((is_wdraw == 'Y') & (~pd.isna(is_wdraw)),
                 np.random.uniform(
                     np.where(np.isnan(wdraw_amt_t0), 0, wdraw_amt_t0),
                     np.where(np.isnan(wdraw_amt), 0, wdraw_amt),
                     size=n_samples
                 ).round(2),
                 0),
        null_prob['wdraw_amt_t3']
    )

    # 修复逻辑运算错误：添加括号并使用逻辑与运算符
    wdraw_amt_t7 = add_nulls(
        np.where((is_wdraw == 'Y') & (~pd.isna(is_wdraw)),
                 np.random.uniform(
                     np.where(np.isnan(wdraw_amt_t3), 0, wdraw_amt_t3),
                     np.where(np.isnan(wdraw_amt), 0, wdraw_amt),
                     size=n_samples
                 ).round(2),
                 0),
        null_prob['wdraw_amt_t7']
    )

    # 修复逻辑运算错误：添加括号并使用逻辑与运算符
    wdraw_amt_t10 = add_nulls(
        np.where((is_wdraw == 'Y') & (~pd.isna(is_wdraw)),
                 np.random.uniform(
                     np.where(np.isnan(wdraw_amt_t7), 0, wdraw_amt_t7),
                     np.where(np.isnan(wdraw_amt), 0, wdraw_amt),
                     size=n_samples
                 ).round(2),
                 0),
        null_prob['wdraw_amt_t10']
    )

    # 修复逻辑运算错误：添加括号并使用逻辑与运算符
    wdraw_amt_t14 = add_nulls(
        np.where((is_wdraw == 'Y') & (~pd.isna(is_wdraw)),
                 np.random.uniform(
                     np.where(np.isnan(wdraw_amt_t10), 0, wdraw_amt_t10),
                     np.where(np.isnan(wdraw_amt), 0, wdraw_amt),
                     size=n_samples
                 ).round(2),
                 0),
        null_prob['wdraw_amt_t14']
    )

    # 修复逻辑运算错误：添加括号并使用逻辑与运算符
    wdraw_amt_t15 = add_nulls(
        np.where((is_wdraw == 'Y') & (~pd.isna(is_wdraw)),
                 np.random.uniform(
                     np.where(np.isnan(wdraw_amt_t14), 0, wdraw_amt_t14),
                     np.where(np.isnan(wdraw_amt), 0, wdraw_amt),
                     size=n_samples
                 ).round(2),
                 0),
        null_prob['wdraw_amt_t15']
    )

    wdraw_amt_t30 = add_nulls(
        wdraw_amt,  # T30等于总提现金额
        null_prob['wdraw_amt_t30']
    )

    # 5. 生成业务系统信息类特征
    # 一级业务线名称
    first_biz_name = add_nulls(
        np.random.choice(['消费信贷', '经营贷款', '信用卡', '其他'], size=n_samples, p=[0.6, 0.2, 0.15, 0.05]),
        null_prob['first_biz_name']
    )

    second_biz_name = add_nulls(
        np.random.choice(['现金贷', '分期贷', '循环贷', '专项贷'], size=n_samples),
        null_prob['second_biz_name']
    )

    third_biz_name = add_nulls(
        np.random.choice(['小额速贷', '大额分期', '应急贷', '消费分期'], size=n_samples),
        null_prob['third_biz_name']
    )

    db_src = add_nulls(
        np.random.choice(['核心系统', '信贷系统', '营销系统', '数据平台'], size=n_samples),
        null_prob['db_src']
    )

    prod_cd = add_nulls(
        [f'P{i:03d}' for i in np.random.randint(1, 50, size=n_samples)],
        null_prob['prod_cd']
    )

    cust_gp_code = add_nulls(
        [f'G{i:04d}' for i in np.random.randint(1, 30, size=n_samples)],
        null_prob['cust_gp_code']
    )

    cust_gp_name = add_nulls(
        np.random.choice(['优质客户', '新客户', '存量客户', '潜力客户', '风险客户'], size=n_samples),
        null_prob['cust_gp_name']
    )

    touch_name = add_nulls(
        [f'触达计划_{i}' for i in np.random.randint(1, 20, size=n_samples)],
        null_prob['touch_name']
    )

    touch_id = add_nulls(
        [f'T{i:05d}' for i in np.random.randint(10000, 99999, size=n_samples)],
        null_prob['touch_id']
    )

    touch_type = add_nulls(
        np.random.choice(['短信', '电话', 'APP推送', '公众号'], size=n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        null_prob['touch_type']
    )

    channel_name = add_nulls(
        np.random.choice(['自有渠道', '合作渠道A', '合作渠道B', '合作渠道C'], size=n_samples),
        null_prob['channel_name']
    )

    channel_id = add_nulls(
        [f'C{i:03d}' for i in np.random.randint(1, 20, size=n_samples)],
        null_prob['channel_id']
    )

    plan_id = add_nulls(
        [f'PL{i:04d}' for i in np.random.randint(1000, 9999, size=n_samples)],
        null_prob['plan_id']
    )

    plan_name = add_nulls(
        [f'营销计划_{i}' for i in np.random.randint(1, 50, size=n_samples)],
        null_prob['plan_name']
    )

    channel_task_id = add_nulls(
        [f'TASK{i:06d}' for i in np.random.randint(100000, 999999, size=n_samples)],
        null_prob['channel_task_id']
    )

    subscribe_no = add_nulls(
        [f'S{i:08d}' for i in np.random.randint(10000000, 99999999, size=n_samples)],
        null_prob['subscribe_no']
    )

    decision_id = add_nulls(
        [f'D{i:05d}' for i in np.random.randint(10000, 99999, size=n_samples)],
        null_prob['decision_id']
    )

    decision_name = add_nulls(
        np.random.choice(['自动审批', '人工审批', '系统初审', '复核'], size=n_samples),
        null_prob['decision_name']
    )

    t1_cust_id = add_nulls(
        [f'CUST{i:09d}' if random.random() > 0.2 else '#N/A' for i in range(n_samples)],
        null_prob['t1_cust_id']
    )

    t2_cust_id = add_nulls(
        [f'USER{i:08d}' if random.random() > 0.3 else '#N/A' for i in range(n_samples)],
        null_prob['t2_cust_id']
    )

    # 组装数据框
    data = {
        # 目标变量
        'aprv_status': labels,

        # 身份与地域信息类
        'age': age,
        'gender_cd': gender_cd,
        'ocr_ethnic': ocr_ethnic,
        'residence_pr_cd': residence_pr_cd,
        'residence_city_cd': residence_city_cd,
        'residence_area_cd': residence_area_cd,
        'census_pr_cd': census_pr_cd,
        'census_city_cd': census_city_cd,
        'census_area_cd': census_area_cd,
        'occ_cd': occ_cd,

        # 信贷核心信息类
        'pril_bal': pril_bal,
        'crdt_lim_yx': crdt_lim_yx,
        'avail_bal_cash': avail_bal_cash,
        'total_loan_cnt': total_loan_cnt,
        'total_loan_amt': total_loan_amt,
        'apply_cnt': apply_cnt,
        'apply_amt': apply_amt,
        'wdraw_cnt': wdraw_cnt,
        'wdraw_amt': wdraw_amt,
        'lim_use_rate': lim_use_rate,

        # 时间与触达信息类
        'cust_recv_time': cust_recv_time,
        'touch_time': touch_time,
        'cust_date': cust_date,
        'push_cnt': push_cnt,
        'sms_charge_cnt': sms_charge_cnt,
        'tel_call_type': tel_call_type,
        'tel_call_dur': tel_call_dur,
        'tel_call_inav_cnt': tel_call_inav_cnt,
        'touch_status': touch_status,
        'click_status': click_status,

        # 行为交互信息类
        'is_coupon_issue': is_coupon_issue,
        'is_credit': is_credit,
        'is_apply': is_apply,
        'is_wdraw': is_wdraw,
        'is_login': is_login,
        'delay_days': delay_days,
        'if_bj_10_yn': if_bj_10_yn,
        'if_bj_30_yn': if_bj_30_yn,
        'wdraw_amt_t0': wdraw_amt_t0,
        'wdraw_amt_t3': wdraw_amt_t3,
        'wdraw_amt_t7': wdraw_amt_t7,
        'wdraw_amt_t10': wdraw_amt_t10,
        'wdraw_amt_t14': wdraw_amt_t14,
        'wdraw_amt_t15': wdraw_amt_t15,
        'wdraw_amt_t30': wdraw_amt_t30,

        # 业务系统信息类
        'first_biz_name': first_biz_name,
        'second_biz_name': second_biz_name,
        'third_biz_name': third_biz_name,
        'db_src': db_src,
        'prod_cd': prod_cd,
        'cust_gp_code': cust_gp_code,
        'cust_gp_name': cust_gp_name,
        'touch_name': touch_name,
        'touch_id': touch_id,
        'touch_type': touch_type,
        'channel_name': channel_name,
        'channel_id': channel_id,
        'plan_id': plan_id,
        'plan_name': plan_name,
        'channel_task_id': channel_task_id,
        'subscribe_no': subscribe_no,
        'decision_id': decision_id,
        'decision_name': decision_name,
        't1_cust_id': t1_cust_id,
        't2_cust_id': t2_cust_id
    }

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 保存为CSV文件
    df.to_csv(output_file, index=False, encoding='utf-8')

    # 统计空值情况
    null_counts = df.isnull().sum()
    null_percent = (null_counts / len(df) * 100).round(2)
    null_stats = pd.DataFrame({'空值数量': null_counts, '空值比例(%)': null_percent})

    return df, null_stats

if __name__ == "__main__":
    # 生成1000行数据
    df, null_stats = generate_credit_data(n_samples=1000, output_file='full_credit_data.csv')

    # 输出空值统计结果
    print(f"成功生成{len(df)}行测试数据，已保存至full_credit_data.csv")
    print("\n空值统计概览：")
    print(f"字段总数：{len(null_stats)}")
    print(f"空值字段数：{sum(null_stats['空值数量'] > 0)}")
    print("\n空值比例最高的5个字段：")
    print(null_stats.sort_values('空值比例(%)', ascending=False).head(5))
    print("\n空值比例最低的5个字段：")
    print(null_stats.sort_values('空值比例(%)').head(5))
