import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 设置随机种子，保证结果可复现
np.random.seed(42)

# 生成基础数据
n_samples = 1000

# 1. 用户基础信息
age = np.random.randint(18, 65, size=n_samples)  # 年龄18-65岁
gender_cd = np.random.choice([1, 2], size=n_samples, p=[0.6, 0.4])  # 1=男，2=女
residence_pr_cd = np.random.choice([11, 31, 44, 45, 32], size=n_samples)  # 模拟省份代码（如11=北京，31=上海）
occ_cd = np.random.choice([101, 102, 103, 104], size=n_samples, p=[0.3, 0.25, 0.2, 0.25])  # 101=企业员工，102=公务员，103=自由职业，104=学生

# 2. 信贷核心指标
crdt_lim_yx = np.random.randint(5000, 50000, size=n_samples)  # 有效额度5000-50000元
pril_bal = np.array([np.random.randint(0, lim) for lim in crdt_lim_yx])  # 在贷余额（不超过额度）
lim_use_ratio = pril_bal / crdt_lim_yx  # 额度使用率

total_loan_cnt = np.random.randint(0, 10, size=n_samples)  # 累计放款次数0-10次
total_loan_amt = np.array([np.random.randint(1000, 10000 * cnt) if cnt > 0 else 0 for cnt in total_loan_cnt])  # 累计放款金额

# 3. 行为交互数据
apply_cnt = np.random.randint(0, 5, size=n_samples)  # 申请次数0-5次
wdraw_cnt = np.array([np.random.randint(0, cnt+1) for cnt in apply_cnt])  # 提现成功次数（≤申请次数）
wdraw_success_rate = np.where(apply_cnt > 0, wdraw_cnt / apply_cnt, 0)  # 提现成功率

if_bj_10_yn = np.random.choice(['Y', 'N'], size=n_samples, p=[0.1, 0.9])  # 近10天被拒
if_bj_30_yn = np.random.choice(['Y', 'N'], size=n_samples, p=[0.2, 0.8])  # 近30天被拒
reject_freq = np.where(if_bj_10_yn == 'Y', 1, 0) + np.where(if_bj_30_yn == 'Y', 1, 0)  # 被拒频率

push_cnt = np.random.randint(0, 5, size=n_samples)  # push推送次数
sms_charge_cnt = np.random.randint(0, 10, size=n_samples)  # 短信条数
is_coupon_issue = np.random.choice(['Y', 'N'], size=n_samples, p=[0.3, 0.7])  # 是否发优惠券

# 4. 时间字段
cust_recv_time = [datetime.now() - timedelta(days=np.random.randint(0, 365)) for _ in range(n_samples)]  # 名单接收日期（近1年）
touch_time = [t + timedelta(hours=np.random.randint(1, 72)) if np.random.random() > 0.2 else None for t in cust_recv_time]  # 触达时间（80%有值）

# 5. 目标变量（授信结果）：结合业务逻辑生成
# 规则：额度使用率低、提现成功率高、被拒频率低的用户更易通过
target = np.where(
    (lim_use_ratio < 0.5) &
    (wdraw_success_rate > 0.6) &
    (reject_freq == 0) &
    (total_loan_cnt > 0),
    1,  # 通过
    np.where(
        (lim_use_ratio > 0.8) |
        (reject_freq >= 1) |
        (wdraw_success_rate < 0.2),
        0,  # 拒绝
        np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])  # 中间地带随机
    )
)

# 6. 组装数据框
sim_data = pd.DataFrame({
    # 基础信息
    'age': age,
    'gender_cd': gender_cd,
    'residence_pr_cd': residence_pr_cd,
    'occ_cd': occ_cd,
    # 信贷核心
    'crdt_lim_yx': crdt_lim_yx,
    'pril_bal': pril_bal,
    'lim_use_ratio': lim_use_ratio,
    'total_loan_cnt': total_loan_cnt,
    'total_loan_amt': total_loan_amt,
    # 行为交互
    'apply_cnt': apply_cnt,
    'wdraw_cnt': wdraw_cnt,
    'wdraw_success_rate': wdraw_success_rate,
    'if_bj_10_yn': if_bj_10_yn,
    'if_bj_30_yn': if_bj_30_yn,
    'reject_freq': reject_freq,
    'push_cnt': push_cnt,
    'sms_charge_cnt': sms_charge_cnt,
    'is_coupon_issue': is_coupon_issue,
    # 时间
    'cust_recv_time': cust_recv_time,
    'touch_time': touch_time,
    # 目标变量
    'target': target
})

# 保存为CSV
sim_data.to_csv('simulated_credit_data.csv', index=False)
print("模拟数据生成完成，保存为 simulated_credit_data.csv")