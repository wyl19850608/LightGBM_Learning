import pandas as pd
import numpy as np
import random
import pickle
import os
from faker import Faker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.exceptions import NotFittedError

# 设置随机种子，保证结果可复现
np.random.seed(42)
random.seed(42)
Faker.seed(42)

# ----------------------
# 1. 数据生成模块
# ----------------------
def generate_simulated_data(n_samples=10000, output_dir='data', output_file='simulated_data.csv'):
    """生成包含多种数据类型的模拟数据，目标变量放在最后一列"""
    os.makedirs(output_dir, exist_ok=True)
    fake = Faker('en_US')

    # 1. 基础数据
    data = {
        'user_id': [f'USER_{i:06d}' for i in range(n_samples)],
        'target_purchase': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    }

    # 2. 连续变量
    age = np.random.normal(loc=35, scale=10, size=n_samples)
    age = np.clip(age, 18, 70).round(1)
    # 随机添加少量缺失值
    age[np.random.choice(n_samples, size=int(n_samples*0.02), replace=False)] = np.nan
    data['age'] = age

    base_income = 3000
    age_related = (age - 18) * 100
    purchase_related = data['target_purchase'] * 2000
    income = base_income + age_related + purchase_related + np.random.normal(0, 1000, n_samples)
    income = np.clip(income, 1000, 15000).round(2)
    # 随机添加少量缺失值
    income[np.random.choice(n_samples, size=int(n_samples*0.01), replace=False)] = np.nan
    data['monthly_income'] = income

    browse_time = np.random.exponential(scale=5, size=n_samples) + data['target_purchase'] * 8
    browse_time = np.clip(browse_time, 0.1, 60).round(2)
    data['browse_time_minutes'] = browse_time

    spend_freq = np.random.normal(loc=3, scale=2, size=n_samples) + data['target_purchase'] * 2
    spend_freq = np.clip(spend_freq, 0.1, 15).round(2)
    data['spending_frequency'] = spend_freq

    # 3. 字符串变量
    genders = ['Male', 'Female', 'Unknown']
    data['gender'] = []
    for purchase in data['target_purchase']:
        prob = [0.45, 0.53, 0.02] if purchase == 1 else [0.55, 0.43, 0.02]
        data['gender'].append(random.choices(genders, weights=prob)[0])

    # 随机添加少量缺失值
    gender_nan_idx = np.random.choice(n_samples, size=int(n_samples*0.03), replace=False)
    for idx in gender_nan_idx:
        data['gender'][idx] = np.nan

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

    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
              'Philadelphia', 'San Antonio', 'San Diego', 'Other City']
    city_probs = [0.15, 0.15, 0.1, 0.1, 0.08, 0.08, 0.07, 0.07, 0.2]
    data['city'] = [random.choices(cities, weights=city_probs)[0] for _ in range(n_samples)]

    educations = ['High School or Below', 'Associate Degree', 'Bachelor Degree', 'Master or Higher']
    data['education'] = []
    for purchase in data['target_purchase']:
        prob = [0.05, 0.2, 0.55, 0.2] if purchase == 1 else [0.15, 0.3, 0.45, 0.1]
        data['education'].append(random.choices(educations, weights=prob)[0])

    # 4. 整数非连续变量
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

    # 5. 布尔变量
    has_coupon = np.random.choice([True, False], size=n_samples, p=[0.3, 0.7])
    data['has_coupon'] = np.where(
        has_coupon,
        np.random.choice([True, False], size=n_samples, p=[0.6, 0.4]),
        data['target_purchase']
    )

    data['is_new_user'] = np.random.choice([True, False], size=n_samples, p=[0.2, 0.8])
    data['joined_promotion'] = np.random.choice([True, False], size=n_samples, p=[0.4, 0.6])
    data['has_notification_enabled'] = np.random.choice([True, False], size=n_samples, p=[0.5, 0.5])

    # 6. 转换为DataFrame并调整列顺序（目标变量放最后）
    df = pd.DataFrame(data)
    string_columns = ['user_id', 'gender', 'occupation', 'city', 'education']
    for col in string_columns:
        df[col] = df[col].astype('string')

    # 调整列顺序，目标变量放最后
    cols = [col for col in df.columns if col != 'target_purchase'] + ['target_purchase']
    df = df[cols]

    # 保存数据
    output_path = os.path.join(output_dir, output_file)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Simulated data generated with {n_samples} records. Saved to: {output_path}")

    return df

# ----------------------
# 2. 有监督分箱模块
# ----------------------
class SupervisedBinner:
    """使用决策树进行有监督分箱"""

    def __init__(self, n_bins=5, min_samples_leaf=0.05, random_state=42):
        self.n_bins = n_bins
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.bins = {}  # 存储每个特征的分箱边界
        self.columns = None  # 需要分箱的特征列名

    def fit(self, X, y, columns=None):
        """训练分箱器"""
        if columns is None:
            self.columns = X.columns.tolist()
        else:
            self.columns = columns

        for col in self.columns:
            X_col = X[[col]].values

            # 使用决策树找到最优分割点
            tree = DecisionTreeClassifier(
                max_leaf_nodes=self.n_bins,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
            tree.fit(X_col, y)

            # 提取并整理分割点
            thresholds = tree.tree_.threshold
            thresholds = thresholds[thresholds != -2]  # 移除无效分割点
            thresholds = np.sort(thresholds)

            # 添加边界值
            self.bins[col] = np.unique(np.concatenate([
                [X_col.min() - 1e-10],
                thresholds,
                [X_col.max() + 1e-10]
            ]))

        return self

    def transform(self, X):
        """应用分箱规则"""
        X_binned = X.copy()
        for col in self.columns:
            if col in self.bins and col in X_binned.columns:
                X_binned[col] = pd.cut(
                    X[col],
                    bins=self.bins[col],
                    labels=False,
                    include_lowest=True
                )
        return X_binned

    def fit_transform(self, X, y, columns=None):
        """训练并应用分箱"""
        self.fit(X, y, columns)
        return self.transform(X)

    def save(self, file_path):
        """保存分箱器"""
        with open(file_path, 'wb') as f:
            pickle.dump({
                'bins': self.bins,
                'columns': self.columns,
                'n_bins': self.n_bins,
                'min_samples_leaf': self.min_samples_leaf,
                'random_state': self.random_state
            }, f)
        print(f"Binner saved to {file_path}")

    @classmethod
    def load(cls, file_path):
        """加载分箱器"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        binner = cls(
            n_bins=data['n_bins'],
            min_samples_leaf=data['min_samples_leaf'],
            random_state=data['random_state']
        )
        binner.bins = data['bins']
        binner.columns = data['columns']
        return binner

# ----------------------
# 3. 数据预处理模块（修复imputer拟合问题）
# ----------------------
def preprocess_data(df, target_col, cat_cols=None, num_cols=None,
                    binner=None, label_encoders=None, imputers=None,
                    final_imputer=None, fit_binner=True, fit_imputers=True):
    """完整的数据预处理流程，确保所有处理器正确拟合"""
    # 分离特征和目标变量
    X = df.drop(columns=[target_col]) if target_col in df.columns else df.copy()
    y = df[target_col] if target_col in df.columns else None

    # 自动识别类别和数值特征
    if cat_cols is None:
        cat_cols = X.select_dtypes(include=['string', 'object', 'category']).columns.tolist()

    if num_cols is None:
        num_cols = [col for col in X.columns if col not in cat_cols]

    # 确保类别特征和数值特征不重复
    cat_cols = [col for col in cat_cols if col in X.columns]
    num_cols = [col for col in num_cols if col in X.columns and col not in cat_cols]

    # 初始化缺失值处理器
    if imputers is None:
        imputers = {
            'categorical': SimpleImputer(strategy='most_frequent'),
            'numerical': SimpleImputer(strategy='median')
        }

    # 初始化最终缺失值处理器
    if final_imputer is None:
        final_imputer = SimpleImputer(strategy='most_frequent')

    # 处理缺失值
    X_imputed = X.copy()

    # 处理类别特征缺失值
    if cat_cols:
        if fit_imputers:
            # 训练类别特征缺失值处理器
            imputers['categorical'].fit(X_imputed[cat_cols])
        # 填充类别特征缺失值
        X_imputed[cat_cols] = imputers['categorical'].transform(X_imputed[cat_cols])

    # 处理数值特征缺失值
    if num_cols:
        if fit_imputers:
            # 训练数值特征缺失值处理器
            imputers['numerical'].fit(X_imputed[num_cols])
        # 填充数值特征缺失值
        X_imputed[num_cols] = imputers['numerical'].transform(X_imputed[num_cols])

    # 处理类别特征 - 标签编码
    if label_encoders is None:
        label_encoders = {}

    X_encoded = X_imputed.copy()
    for col in cat_cols:
        # 确保没有缺失值（已通过imputer处理）
        X_encoded[col] = X_encoded[col].fillna('Unknown')

        if col not in label_encoders:
            # 确保编码器训练时包含'Unknown'类别
            unique_vals = X_encoded[col].unique()
            if 'Unknown' not in unique_vals:
                temp_vals = np.append(unique_vals, 'Unknown')
            else:
                temp_vals = unique_vals

            le = LabelEncoder()
            le.fit(temp_vals)
            label_encoders[col] = le

        # 处理未见过的类别
        le = label_encoders[col]
        known_classes = set(le.classes_)
        X_encoded[col] = X_encoded[col].apply(
            lambda x: x if x in known_classes else 'Unknown'
        )
        X_encoded[col] = le.transform(X_encoded[col])

    # 处理数值特征 - 有监督分箱
    if fit_binner:
        binner = SupervisedBinner(n_bins=5, min_samples_leaf=0.05)
        X_binned = binner.fit_transform(X_encoded[num_cols], y)
    else:
        if binner is None:
            raise ValueError("When fit_binner is False, a trained binner must be provided")
        X_binned = binner.transform(X_encoded[num_cols])

    # 合并处理后的特征
    X_processed = pd.concat([
        X_encoded[cat_cols],
        X_binned[num_cols]
    ], axis=1)

    # 再次检查并处理可能的缺失值（分箱过程中可能产生）
    if fit_imputers:
        # 确保final_imputer已拟合
        final_imputer.fit(X_processed)
    else:
        # 验证final_imputer是否已拟合
        try:
            # 尝试一个简单的检查来验证是否已拟合
            if not hasattr(final_imputer, 'statistics_'):
                raise NotFittedError("final_imputer has not been fitted yet.")
        except NotFittedError:
            raise NotFittedError("final_imputer is not fitted. Call 'fit' before using 'transform'.")

    # 使用final_imputer转换数据
    X_processed = pd.DataFrame(
        final_imputer.transform(X_processed),
        columns=X_processed.columns,
        index=X_processed.index
    )

    return X_processed, y, binner, label_encoders, imputers, cat_cols, num_cols, final_imputer

# ----------------------
# 4. 模型训练与保存模块
# ----------------------
def train_and_save_model(train_data_path, target_col, model_dir='model_files'):
    """训练模型并保存所有组件"""
    os.makedirs(model_dir, exist_ok=True)

    # 1. 加载训练数据
    print("Loading training data...")
    df = pd.read_csv(train_data_path)
    print(f"Training data shape: {df.shape}")
    print(f"Missing values in training data:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"Target variable '{target_col}' distribution:")
    print(df[target_col].value_counts(normalize=True).round(4))

    # 2. 分割训练集和验证集
    print("Splitting into training and validation sets...")
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[target_col]
    )

    # 3. 预处理训练数据（包含缺失值处理）
    print("Preprocessing training data (including missing value handling)...")
    X_train, y_train, binner, label_encoders, imputers, cat_cols, num_cols, final_imputer = preprocess_data(
        train_df,
        target_col=target_col,
        fit_binner=True,
        fit_imputers=True
    )

    # 4. 训练模型
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 5. 在验证集上评估模型
    print("Evaluating model on validation set...")
    X_val, y_val, _, _, _, _, _, _ = preprocess_data(
        val_df,
        target_col=target_col,
        cat_cols=cat_cols,
        num_cols=num_cols,
        binner=binner,
        label_encoders=label_encoders,
        imputers=imputers,
        final_imputer=final_imputer,  # 传入已拟合的final_imputer
        fit_binner=False,
        fit_imputers=False
    )

    # 预测
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # 评估指标
    print("\nValidation set metrics:")
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_val, y_pred_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

    # 6. 保存模型和所有组件
    model_path = os.path.join(model_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    binner.save(os.path.join(model_dir, 'binner.pkl'))

    with open(os.path.join(model_dir, 'label_encoders.pkl'), 'wb') as f:
        pickle.dump(label_encoders, f)

    with open(os.path.join(model_dir, 'imputers.pkl'), 'wb') as f:
        pickle.dump(imputers, f)

    with open(os.path.join(model_dir, 'final_imputer.pkl'), 'wb') as f:
        pickle.dump(final_imputer, f)

    with open(os.path.join(model_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump({
            'cat_cols': cat_cols,
            'num_cols': num_cols,
            'target_col': target_col
        }, f)

    print(f"\nAll model components saved to {model_dir} directory")
    return model

# ----------------------
# 5. 模型加载与预测模块
# ----------------------
def load_model_and_predict(new_data_path, model_dir='model_files', output_path=None):
    """加载模型和预处理组件，对新数据进行预测"""
    # 1. 加载模型组件
    print("Loading model components...")
    model_path = os.path.join(model_dir, 'model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    binner = SupervisedBinner.load(os.path.join(model_dir, 'binner.pkl'))

    with open(os.path.join(model_dir, 'label_encoders.pkl'), 'rb') as f:
        label_encoders = pickle.load(f)

    with open(os.path.join(model_dir, 'imputers.pkl'), 'rb') as f:
        imputers = pickle.load(f)

    with open(os.path.join(model_dir, 'final_imputer.pkl'), 'rb') as f:
        final_imputer = pickle.load(f)

    with open(os.path.join(model_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    cat_cols = metadata['cat_cols']
    num_cols = metadata['num_cols']
    target_col = metadata['target_col']

    # 2. 加载新数据
    print("Loading new data...")
    new_data = pd.read_csv(new_data_path)
    print(f"New data shape: {new_data.shape}")
    print(f"Missing values in new data:\n{new_data.isnull().sum()[new_data.isnull().sum() > 0]}")

    # 3. 预处理新数据
    print("Preprocessing new data...")
    X_processed, _, _, _, _, _, _, _ = preprocess_data(
        new_data,
        target_col=target_col,
        cat_cols=cat_cols,
        num_cols=num_cols,
        binner=binner,
        label_encoders=label_encoders,
        imputers=imputers,
        final_imputer=final_imputer,  # 传入已拟合的final_imputer
        fit_binner=False,
        fit_imputers=False
    )

    # 4. 预测
    print("Making predictions...")
    y_pred = model.predict(X_processed)
    y_pred_proba = model.predict_proba(X_processed)[:, 1]

    # 5. 整理结果
    result = new_data.copy()
    result['predicted_label'] = y_pred
    result['predicted_probability'] = y_pred_proba.round(4)

    # 如果新数据包含目标变量，计算评估指标
    if target_col in result.columns:
        print("\nNew data metrics:")
        print(f"Accuracy: {accuracy_score(result[target_col], y_pred):.4f}")
        print(f"AUC: {roc_auc_score(result[target_col], y_pred_proba):.4f}")

    # 6. 保存结果
    if output_path:
        result.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Predictions saved to {output_path}")

    return result

# ----------------------
# 6. 主函数（流程控制）
# ----------------------
def main():
    # 配置参数
    n_samples = 10000
    data_dir = 'data'
    model_dir = 'model_files'
    train_data_file = 'simulated_data.csv'
    new_data_file = 'new_simulated_data.csv'
    target_col = 'target_purchase'
    predictions_file = 'predictions.csv'

    # 步骤1: 生成模拟训练数据
    print("===== Step 1: Generating simulated training data =====")
    train_data_path = os.path.join(data_dir, train_data_file)
    generate_simulated_data(n_samples=n_samples, output_dir=data_dir, output_file=train_data_file)

    # 步骤2: 生成新数据（用于预测）
    print("\n===== Step 2: Generating new data for prediction =====")
    new_data_path = os.path.join(data_dir, new_data_file)
    generate_simulated_data(n_samples=2000, output_dir=data_dir, output_file=new_data_file)

    # 步骤3: 训练模型并保存
    print("\n===== Step 3: Training model =====")
    train_and_save_model(train_data_path, target_col, model_dir)

    # 步骤4: 加载模型并预测新数据
    print("\n===== Step 4: Predicting on new data =====")
    predictions = load_model_and_predict(
        new_data_path,
        model_dir,
        output_path=os.path.join(data_dir, predictions_file)
    )

    # 显示部分预测结果
    print("\nSample predictions:")
    print(predictions[[target_col, 'predicted_label', 'predicted_probability']].head(10))

if __name__ == "__main__":
    main()
