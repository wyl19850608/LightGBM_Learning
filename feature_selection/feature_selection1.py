import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, chi2, f_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Union, Optional
import warnings
warnings.filterwarnings('ignore')

class FeatureSelector:
    """特征选择工具，集成统计分析和模型反馈方法"""

    def __init__(self, data: pd.DataFrame, target: str,
                 categorical_features: List[str] = None,
                 numerical_features: List[str] = None):
        """
        初始化特征选择器

        参数:
            data: 包含特征和目标变量的DataFrame
            target: 目标变量名称
            categorical_features: 分类特征列表
            numerical_features: 数值特征列表
        """
        self.data = data.copy()
        self.target = target
        self.categorical_features = categorical_features or []
        self.numerical_features = numerical_features or []

        # 自动识别特征类型(如果未指定)
        if not self.categorical_features and not self.numerical_features:
            self._identify_feature_types()

        # 检查特征是否存在
        self._validate_features()

        # 存储结果
        self.feature_importance = {}
        self.redundant_features = []
        self.selected_features = []

    def _identify_feature_types(self):
        """自动识别特征类型"""
        for col in self.data.columns:
            if col == self.target:
                continue

            # 如果是object类型或唯一值数量少于20，视为分类特征
            if self.data[col].dtype == 'object' or self.data[col].nunique() < 20:
                self.categorical_features.append(col)
            else:
                self.numerical_features.append(col)

    def _validate_features(self):
        """验证特征是否存在于数据中"""
        all_features = self.categorical_features + self.numerical_features
        missing_features = [f for f in all_features if f not in self.data.columns]

        if missing_features:
            raise ValueError(f"特征不存在于数据中: {missing_features}")

        if self.target not in self.data.columns:
            raise ValueError(f"目标变量 {self.target} 不存在于数据中")

    def handle_missing_values(self, threshold: float = 0.8) -> None:
        """
        处理缺失值

        参数:
            threshold: 删除缺失率超过此阈值的特征
        """
        missing_ratio = self.data.isnull().sum() / len(self.data)

        # 删除缺失率过高的特征
        for feature, ratio in missing_ratio.items():
            if ratio > threshold and feature != self.target:
                if feature in self.categorical_features:
                    self.categorical_features.remove(feature)
                if feature in self.numerical_features:
                    self.numerical_features.remove(feature)
                self.data.drop(feature, axis=1, inplace=True)
                print(f"删除缺失率高的特征: {feature} ({ratio:.2%})")

        # 填充剩余缺失值
        for feature in self.categorical_features:
            self.data[feature].fillna(self.data[feature].mode()[0], inplace=True)

        for feature in self.numerical_features:
            self.data[feature].fillna(self.data[feature].median(), inplace=True)

    def analyze_feature_distribution(self) -> Dict[str, float]:
        """分析特征分布，识别低方差特征"""
        low_variance_features = {}

        for feature in self.categorical_features:
            # 计算最频繁类别的占比
            top_class_ratio = self.data[feature].value_counts(normalize=True).max()
            if top_class_ratio > 0.95:  # 如果某类别占比超过95%
                low_variance_features[feature] = top_class_ratio

        for feature in self.numerical_features:
            # 计算变异系数(标准差/均值)
            if self.data[feature].std() == 0:
                low_variance_features[feature] = 1.0
            else:
                cv = self.data[feature].std() / abs(self.data[feature].mean())
                if cv < 0.01:  # 如果变异系数过小
                    low_variance_features[feature] = cv

        return low_variance_features

    def remove_low_variance_features(self) -> None:
        """移除低方差特征"""
        low_variance_features = self.analyze_feature_distribution()

        for feature in low_variance_features.keys():
            if feature in self.categorical_features:
                self.categorical_features.remove(feature)
            if feature in self.numerical_features:
                self.numerical_features.remove(feature)
            self.data.drop(feature, axis=1, inplace=True)
            print(f"删除低方差特征: {feature}")

    def calculate_feature_correlation(self, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        """
        计算特征相关性，识别冗余特征

        参数:
            threshold: 相关系数阈值，超过此值视为高度相关

        返回:
            高度相关的特征对列表
        """
        # 对分类特征进行编码
        encoded_data = self.data.copy()
        for feature in self.categorical_features:
            le = LabelEncoder()
            encoded_data[feature] = le.fit_transform(encoded_data[feature])

        # 计算相关系数矩阵
        corr_matrix = encoded_data.corr().abs()

        # 找出高度相关的特征对
        highly_correlated = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    col_i = corr_matrix.columns[i]
                    col_j = corr_matrix.columns[j]
                    if col_i != self.target and col_j != self.target:
                        highly_correlated.append((col_i, col_j, corr_matrix.iloc[i, j]))

        return highly_correlated

    def remove_redundant_features(self, threshold: float = 0.8) -> None:
        """
        移除冗余特征

        参数:
            threshold: 相关系数阈值，超过此值视为高度相关
        """
        highly_correlated = self.calculate_feature_correlation(threshold)

        # 基于特征重要性决定保留哪个特征
        features_to_remove = set()

        # 如果已经计算了特征重要性，则使用它
        if self.feature_importance:
            for feat1, feat2, _ in highly_correlated:
                if feat1 not in features_to_remove and feat2 not in features_to_remove:
                    # 保留更重要的特征
                    if self.feature_importance.get(feat1, 0) > self.feature_importance.get(feat2, 0):
                        features_to_remove.add(feat2)
                    else:
                        features_to_remove.add(feat1)
        else:
            # 否则随机选择一个保留
            for feat1, feat2, _ in highly_correlated:
                if feat1 not in features_to_remove and feat2 not in features_to_remove:
                    # 选择唯一值更多的特征(通常更有信息量)
                    if self.data[feat1].nunique() > self.data[feat2].nunique():
                        features_to_remove.add(feat2)
                    else:
                        features_to_remove.add(feat1)

        # 执行删除
        for feature in features_to_remove:
            if feature in self.categorical_features:
                self.categorical_features.remove(feature)
            if feature in self.numerical_features:
                self.numerical_features.remove(feature)
            self.data.drop(feature, axis=1, inplace=True)
            self.redundant_features.append(feature)
            print(f"删除冗余特征: {feature}")

    def calculate_statistical_importance(self) -> None:
        """
        计算统计特征重要性
        - 分类特征: 使用卡方检验和互信息
        - 数值特征: 使用F检验和互信息
        """
        # 对分类特征进行编码
        encoded_data = self.data.copy()
        for feature in self.categorical_features:
            le = LabelEncoder()
            encoded_data[feature] = le.fit_transform(encoded_data[feature])

        # 准备特征和目标
        X = encoded_data[self.categorical_features + self.numerical_features]
        y = encoded_data[self.target]

        # 初始化重要性分数
        self.feature_importance = {feature: 0 for feature in X.columns}

        # 计算分类特征的卡方检验和互信息
        if self.categorical_features:
            chi2_scores, _ = chi2(X[self.categorical_features], y)
            mi_scores = mutual_info_classif(X[self.categorical_features], y, discrete_features=True)

            for i, feature in enumerate(self.categorical_features):
                # 综合卡方检验和互信息的分数
                self.feature_importance[feature] = (chi2_scores[i] / chi2_scores.max() +
                                                    mi_scores[i] / mi_scores.max()) / 2

        # 计算数值特征的F检验和互信息
        if self.numerical_features:
            f_scores, _ = f_classif(X[self.numerical_features], y)
            mi_scores = mutual_info_classif(X[self.numerical_features], y, discrete_features=False)

            for i, feature in enumerate(self.numerical_features):
                # 综合F检验和互信息的分数
                self.feature_importance[feature] = (f_scores[i] / f_scores.max() +
                                                    mi_scores[i] / mi_scores.max()) / 2

    def calculate_model_importance(self) -> None:
        """使用随机森林计算特征重要性"""
        # 对分类特征进行编码
        encoded_data = self.data.copy()
        for feature in self.categorical_features:
            le = LabelEncoder()
            encoded_data[feature] = le.fit_transform(encoded_data[feature])

        # 标准化数值特征
        scaler = StandardScaler()
        encoded_data[self.numerical_features] = scaler.fit_transform(
            encoded_data[self.numerical_features])

        # 准备特征和目标
        X = encoded_data[self.categorical_features + self.numerical_features]
        y = encoded_data[self.target]

        # 训练随机森林模型
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)

        # 保存特征重要性
        for i, feature in enumerate(X.columns):
            self.feature_importance[feature] = rf.feature_importances_[i]

    def plot_feature_importance(self, top_n: int = 20) -> None:
        """
        绘制特征重要性图

        参数:
            top_n: 显示前多少个重要特征
        """
        if not self.feature_importance:
            raise ValueError("请先计算特征重要性")

        # 排序并取前top_n个特征
        sorted_importance = sorted(self.feature_importance.items(),
                                   key=lambda x: x[1], reverse=True)[:top_n]

        features, importance = zip(*sorted_importance)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(importance), y=list(features))
        plt.title(f"Top {top_n} Feature Importance")
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.show()

    def select_features_by_importance(self, threshold: float = 0.01) -> None:
        """
        基于重要性阈值选择特征

        参数:
            threshold: 重要性阈值，低于此值的特征将被剔除
        """
        if not self.feature_importance:
            raise ValueError("请先计算特征重要性")

        # 选择重要性大于阈值的特征
        self.selected_features = [feature for feature, score in self.feature_importance.items()
                                  if score > threshold]

        # 更新特征列表
        self.categorical_features = [f for f in self.categorical_features
                                     if f in self.selected_features]
        self.numerical_features = [f for f in self.numerical_features
                                   if f in self.selected_features]

        print(f"基于重要性选择了 {len(self.selected_features)} 个特征")

    def get_selected_data(self) -> pd.DataFrame:
        """获取选择后的数据集"""
        all_features = self.categorical_features + self.numerical_features
        return self.data[all_features + [self.target]]

    def run_full_pipeline(self,
                          missing_threshold: float = 0.8,
                          correlation_threshold: float = 0.8,
                          importance_threshold: float = 0.01) -> None:
        """
        运行完整的特征选择流程

        参数:
            missing_threshold: 缺失值阈值
            correlation_threshold: 相关系数阈值
            importance_threshold: 重要性阈值
        """
        print("开始特征选择流程...")

        # 1. 处理缺失值
        self.handle_missing_values(missing_threshold)

        # 2. 移除低方差特征
        self.remove_low_variance_features()

        # 3. 计算并移除冗余特征
        self.calculate_statistical_importance()  # 先计算重要性，用于决定保留哪个特征
        self.remove_redundant_features(correlation_threshold)

        # 4. 计算特征重要性
        self.calculate_model_importance()

        # 5. 基于重要性选择特征
        self.select_features_by_importance(importance_threshold)

        print(f"特征选择完成! 初始特征数: {len(self.data.columns) - 1}, "
              f"最终选择特征数: {len(self.selected_features)}")


# 示例用法
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    n_samples = 100

    data = pd.DataFrame({
        'user_age': np.random.normal(30, 10, n_samples),
        'user_gender': np.random.choice(['M', 'F', 'O'], n_samples),
        'user_income': np.random.lognormal(10, 0.5, n_samples),
        'user_click_history': np.random.poisson(5, n_samples),
        'item_price': np.random.lognormal(5, 0.5, n_samples),
        'item_category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'item_rating': np.random.uniform(1, 5, n_samples),
        'is_promotion': np.random.choice([0, 1], n_samples),
        'day_of_week': np.random.choice(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], n_samples),
        'hour_of_day': np.random.randint(0, 24, n_samples),
        'random_feature': np.random.random(n_samples),  # 随机噪声特征
        'redundant_feature1': np.random.normal(30, 10, n_samples),  # 冗余特征
        'redundant_feature2': np.random.normal(30, 10, n_samples) + np.random.normal(0, 0.1, n_samples),  # 冗余特征
        'target': np.random.binomial(1, 0.5, n_samples)  # 目标变量
    })

    # 创建一些缺失值
    for col in data.columns:
        if col != 'target':
            if col in ['user_age', 'user_income']:
                continue
            mask = np.random.random(n_samples) < 0.9
            data.loc[mask, col] = np.nan


    # 统计每列中各个数值（包括空值）的出现次数
    for column in data.columns:
        print(f"列 {column} 的数值出现次数（包含空值）：")
        value_counts = data[column].value_counts(dropna=False)
        print(value_counts)
        print("\n")


    # 初始化特征选择器
    fs = FeatureSelector(
        data=data,
        target='target',
        categorical_features=['user_gender', 'item_category', 'is_promotion', 'day_of_week'],
        numerical_features=['user_age', 'user_income', 'user_click_history', 'item_price',
                            'item_rating', 'hour_of_day', 'random_feature',
                            'redundant_feature1', 'redundant_feature2']
    )

    # 运行完整的特征选择流程
    fs.run_full_pipeline(
        missing_threshold=0.8,
        correlation_threshold=0.8,
        importance_threshold=0.01
    )


    # 显示特征重要性图
    fs.plot_feature_importance(top_n=10)

    # 获取选择后的数据集
    selected_data = fs.get_selected_data()
    print("\n选择后的特征:")
    print(selected_data.columns.tolist())

    for column in selected_data.columns:
        print(f"列 {column} 的数值出现次数（包含空值）：")
        value_counts = selected_data[column].value_counts(dropna=False)
        print(value_counts)
        print("\n")
