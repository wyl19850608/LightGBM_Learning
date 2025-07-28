import pandas as pd
import numpy as np
import random
from faker import Faker
import os

# 初始化随机种子，保证结果可复现
np.random.seed(42)
random.seed(42)

# 初始化Faker生成英文虚假数据
fake = Faker('en_US')
Faker.seed(42)

def generate_simulated_data(n_samples=10000, output_dir='data', output_file='simulated_data_string_type.csv'):
    """
    生成模拟数据并将字符串列显式转换为string类型（而非默认的object）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------
    # 1. 生成基础数据
    # ----------------------
    data = {
        'user_id': [f'USER_{i:06d}' for i in range(n_samples)],
        'target_purchase': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    }

    # ----------------------
    # 2. 连续变量
    # ----------------------
    age = np.random.normal(loc=35, scale=10, size=n_samples)
    age = np.clip(age, 18, 70).round(1)
    data['age'] = age

    base_income = 3000
    age_related = (age - 18) * 100
    purchase_related = data['target_purchase'] * 2000
    income = base_income + age_related + purchase_related + np.random.normal(0, 1000, n_samples)
    income = np.clip(income, 1000, 15000).round(2)
    data['monthly_income'] = income

    browse_time = np.random.exponential(scale=5, size=n_samples) + data['target_purchase'] * 8
    browse_time = np.clip(browse_time, 0.1, 60).round(2)
    data['browse_time_minutes'] = browse_time

    spend_freq = np.random.normal(loc=3, scale=2, size=n_samples) + data['target_purchase'] * 2
    spend_freq = np.clip(spend_freq, 0.1, 15).round(2)
    data['spending_frequency'] = spend_freq

    # ----------------------
    # 3. 字符串变量（后续会显式转为string类型）
    # ----------------------
    # 性别
    genders = ['Male', 'Female', 'Unknown']
    data['gender'] = []
    for purchase in data['target_purchase']:
        prob = [0.45, 0.53, 0.02] if purchase == 1 else [0.55, 0.43, 0.02]
        data['gender'].append(random.choices(genders, weights=prob)[0])

    # 职业
    occupations = [
        'Student', 'Corporate Employee', 'Teacher', 'Doctor', 'Engineer',
        'Freelancer', 'Civil Servant', 'Salesperson', 'Manager', 'Other'
    ]
    data['occupation'] = []
    for a, t in zip(age, data['target_purchase']):
        if a < 25:
            probs = [0.6, 0.1, 0.05, 0.02, 0.08, 0.05, 0.02, 0.03, 0.01, 0.04]
        elif a < 40:
            probs = [0.05, 0.3, 0.08, 0.05, 0.2, 0.08, 0.05, 0.08, 0.05, 0.06]
        else:
            probs = [0.01, 0.2, 0.15, 0.1, 0.1, 0.05, 0.15, 0.08, 0.1, 0.06]

        if t == 1:
            probs = [p * 1.2 if i in [1, 7, 8] else p for i, p in enumerate(probs)]
            probs = [p / sum(probs) for p in probs]

        data['occupation'].append(random.choices(occupations, weights=probs)[0])

    # 城市
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
              'Philadelphia', 'San Antonio', 'San Diego', 'Other City']
    city_probs = [0.15, 0.15, 0.1, 0.1, 0.08, 0.08, 0.07, 0.07, 0.2]
    data['city'] = [random.choices(cities, weights=city_probs)[0] for _ in range(n_samples)]

    # 教育程度
    educations = ['High School or Below', 'Associate Degree', 'Bachelor Degree', 'Master or Higher']
    data['education'] = []
    for purchase in data['target_purchase']:
        prob = [0.05, 0.2, 0.55, 0.2] if purchase == 1 else [0.15, 0.3, 0.45, 0.1]
        data['education'].append(random.choices(educations, weights=prob)[0])

    # ----------------------
    # 4. 整数非连续变量
    # ----------------------
    family_sizes = np.random.randint(1, 9, size=n_samples)
    family_sizes = np.where(
        data['target_purchase'] == 1,
        np.clip(family_sizes + np.random.choice([0, 1], size=n_samples), 1, 8),
        family_sizes
    )
    data['family_size'] = family_sizes

    login_counts = np.random.poisson(lam=5, size=n_samples)
    login_counts = np.clip(login_counts, 0, 20)
    login_counts = login_counts + data['target_purchase'] * np.random.randint(1, 5, size=n_samples)
    login_counts = np.clip(login_counts, 0, 20)
    data['login_count_30d'] = login_counts

    member_levels = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.4, 0.3, 0.15, 0.1, 0.05])
    data['member_level'] = np.where(
        data['target_purchase'] == 1,
        np.clip(member_levels + np.random.choice([0, 1, 2], size=n_samples), 1, 5),
        member_levels
    )

    category_counts = np.random.randint(1, 11, size=n_samples)
    category_counts = np.where(
        data['target_purchase'] == 1,
        np.clip(category_counts + np.random.randint(1, 4, size=n_samples), 1, 10),
        category_counts
    )
    data['browsed_categories'] = category_counts

    # ----------------------
    # 5. 布尔变量
    # ----------------------
    has_coupon = np.random.choice([True, False], size=n_samples, p=[0.3, 0.7])
    data['has_coupon'] = np.where(
        has_coupon,
        np.random.choice([True, False], size=n_samples, p=[0.6, 0.4]),
        data['target_purchase']
    )

    data['is_new_user'] = np.random.choice([True, False], size=n_samples, p=[0.2, 0.8])
    data['joined_promotion'] = np.random.choice([True, False], size=n_samples, p=[0.4, 0.6])
    data['has_notification_enabled'] = np.random.choice([True, False], size=n_samples, p=[0.5, 0.5])

    # ----------------------
    # 6. 转换为DataFrame并显式设置string类型
    # ----------------------
    df = pd.DataFrame(data)

    # 关键修改：将字符串列显式转换为string类型（而非默认的object）
    string_columns = ['user_id', 'gender', 'occupation', 'city', 'education']
    for col in string_columns:
        df[col] = df[col].astype('string')  # 显式转换为Pandas的StringDtype

    # 调整列顺序
    cols = ['user_id', 'target_purchase'] + [col for col in df.columns if col not in ['user_id', 'target_purchase']]
    df = df[cols]

    # 保存数据
    output_path = os.path.join(output_dir, output_file)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Simulated data generated with {n_samples} records. Saved to: {output_path}")

    return df

def analyze_simulated_data(df):
    """分析数据类型分布"""
    print("\n===== Data Type Distribution =====")
    print(df.dtypes)  # 此时字符串列应为string类型而非object

    print("\n===== First 5 Rows Preview =====")
    print(df.head())

    # 验证string类型是否设置成功
    string_cols = [col for col in df.columns if pd.api.types.is_string_dtype(df[col])]
    print("\n===== Columns with String Type =====")
    print(string_cols)

if __name__ == "__main__":
    df = generate_simulated_data(n_samples=10000)
    analyze_simulated_data(df)
