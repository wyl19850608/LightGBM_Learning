import pandas as pd
import numpy as np
import json
import os
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import chi2_contingency

class AdvancedFeatureBinner:
    """高级连续特征分箱处理器，支持为每个个特征配置多种分箱方案"""

    def __init__(self):
        self.binning_config = {}  # 存储分箱配置，结构: {特征: {分箱方案ID: 配置详情}}
        self.features = []        # 所有分箱特征列表

    def _generate_bin_id(self, method, n_bins):
        """生成分箱方案唯一标识"""
        return f"{method}_{n_bins}"

    def _generate_feature_column_name(self, feature, bin_id):
        """生成新特征列名，格式: 原始特征_method_n_bins"""
        return f"{feature}_{bin_id}"

    def fit(self, df, features_config, target=None, min_samples_leaf=0.05, random_state=42):
        """
        训练分箱规则，支持为每个特征配置多种分箱方案

        参数:
            df: 包含特征的数据框
            features_config: 字典形式，键为特征名，值为分箱方案列表
                      例如: {
                          'age': [
                              {'method': 'equal_width', 'n_bins': 3, 'replace_na': -1},
                              {'method': 'kmeans', 'n_bins': 4}
                          ],
                          'income': [
                              {'method': 'equal_freq', 'n_bins': 5, 'replace_na': 999}
                          ]
                      }
                      支持的分箱方法: 'equal_width', 'equal_freq', 'kmeans', 'chi2', 'decision_tree'
                      可选参数: replace_na - 该特征的缺失值替换值，默认为-1
            target: 目标变量（有监督分箱时需要）
            min_samples_leaf: 决策树分箱时的最小叶子节点样本比例（全局参数）
            random_state: 随机种子
        """
        # 验证配置参数格式
        if not isinstance(features_config, dict):
            raise ValueError("features_config参数必须是字典类型，格式为{特征名: [分箱方案列表]}")

        self.features = list(features_config.keys())
        self.binning_config = {feature: {} for feature in self.features}

        for feature, bin_strategies in features_config.items():
            if feature not in df.columns:
                raise ValueError(f"特征 {feature} 不在数据框中")

            if not isinstance(bin_strategies, list) or len(bin_strategies) == 0:
                raise ValueError(f"特征 {feature} 的分箱方案必须是非空列表")

            # 提取特征值（排除NaN）
            values = df[feature].dropna().values.reshape(-1, 1)

            # 为每个分箱方案训练
            for strategy in bin_strategies:
                # 验证分箱方案参数
                if not isinstance(strategy, dict) or 'method' not in strategy or 'n_bins' not in strategy:
                    raise ValueError(f"特征 {feature} 的分箱方案配置不正确，必须包含'method'和'n_bins'")

                method = strategy['method']
                n_bins = strategy['n_bins']
                replace_na = strategy.get('replace_na', -1)

                # 生成分箱方案ID
                bin_id = self._generate_bin_id(method, n_bins)

                # 存储基本配置
                self.binning_config[feature][bin_id] = {
                    'method': method,
                    'n_bins': n_bins,
                    'replace_na': replace_na
                }

                # 等宽分箱
                if method == 'equal_width':
                    bins = np.linspace(values.min(), values.max(), n_bins + 1)
                    self.binning_config[feature][bin_id]['bins'] = bins.tolist()

                # 等频分箱
                elif method == 'equal_freq':
                    quantiles = np.linspace(0, 1, n_bins + 1)
                    bins = np.percentile(values, quantiles * 100)
                    # 处理重复值导致的边界问题
                    bins = np.unique(bins)
                    # 确保分箱数量正确
                    if len(bins) < n_bins + 1:
                        bins = np.linspace(values.min(), values.max(), n_bins + 1)
                    self.binning_config[feature][bin_id]['bins'] = bins.tolist()

                # K-means聚类分箱
                elif method == 'kmeans':
                    kmeans = KMeans(n_clusters=n_bins, random_state=random_state)
                    kmeans.fit(values)
                    bins = np.sort(kmeans.cluster_centers_.flatten())
                    # 扩展边界
                    bins = np.concatenate([
                        [values.min() - 1e-10],
                        (bins[:-1] + bins[1:]) / 2,  # 取中心点之间的中点作为边界
                        [values.max() + 1e-10]
                    ])
                    self.binning_config[feature][bin_id]['bins'] = bins.tolist()

                # 决策树分箱（有监督）
                elif method == 'decision_tree':
                    if target is None:
                        raise ValueError(f"特征 {feature} 使用决策树分箱需要指定目标变量target")

                    y = df[target].loc[df[feature].notna()].values
                    tree = DecisionTreeClassifier(
                        max_leaf_nodes=n_bins,
                        min_samples_leaf=min_samples_leaf,
                        random_state=random_state
                    )
                    tree.fit(values, y)

                    # 提取决策树的分割点
                    thresholds = tree.tree_.threshold
                    thresholds = thresholds[thresholds != -2]  # 排除无效分割点
                    thresholds = np.sort(thresholds)

                    # 添加边界值
                    bins = np.concatenate([
                        [values.min() - 1e-10],
                        thresholds,
                        [values.max() + 1e-10]
                    ])
                    self.binning_config[feature][bin_id]['bins'] = bins.tolist()

                # 卡方分箱（有监督）
                elif method == 'chi2':
                    if target is None:
                        raise ValueError(f"特征 {feature} 使用卡方分箱需要指定目标变量target")

                    # 初始化为每个值一个区间
                    unique_vals = np.sort(np.unique(values))
                    bins = unique_vals.tolist()

                    # 合并区间直到达到指定的分箱数量
                    while len(bins) > n_bins:
                        min_chi = float('inf')
                        merge_idx = 0

                        # 计算相邻区间的卡方值
                        for i in range(len(bins) - 1):
                            # 创建两个区间
                            bin1 = (values >= bins[i]) & (values < bins[i+1])
                            bin2 = (values >= bins[i+1]) & (values < bins[i+2] if i+2 < len(bins) else values <= bins[i+1])

                            # 创建列联表
                            observed = np.array([
                                [np.sum(bin1 & (y == 0)), np.sum(bin1 & (y == 1))],
                                [np.sum(bin2 & (y == 0)), np.sum(bin2 & (y == 1))]
                            ])

                            # 计算卡方值
                            chi2, _, _, _ = chi2_contingency(observed)

                            if chi2 < min_chi:
                                min_chi = chi2
                                merge_idx = i

                        # 合并卡方值最小的相邻区间
                        bins.pop(merge_idx + 1)

                    # 添加边界值
                    bins = np.concatenate([
                        [values.min() - 1e-10],
                        bins,
                        [values.max() + 1e-10]
                    ])
                    self.binning_config[feature][bin_id]['bins'] = bins.tolist()

                else:
                    raise ValueError(f"特征 {feature} 使用了不支持的分箱方法: {method}")

                # 计算每个分箱的统计信息
                self._calculate_bin_stats(df, feature, bin_id, target)

        return self

    def _calculate_bin_stats(self, df, feature, bin_id, target):
        """计算每个分箱方案的统计信息"""
        bins = self.binning_config[feature][bin_id]['bins']
        bin_labels = list(range(len(bins) - 1))

        # 创建分箱列
        df_temp = df[[feature]].copy()
        df_temp['bin'] = pd.cut(
            df_temp[feature],
            bins=bins,
            labels=bin_labels,
            include_lowest=True
        )

        # 计算基本统计信息
        bin_stats = df_temp.groupby('bin')[feature].agg(['count', 'min', 'max', 'mean']).to_dict('index')

        # 如果有目标变量，计算目标变量相关统计
        if target and target in df.columns:
            df_temp[target] = df[target]
            target_stats = df_temp.groupby('bin')[target].agg(['mean', 'sum']).to_dict('index')
            for bin_idx in bin_stats:
                bin_stats[bin_idx].update({
                    f'{target}_mean': target_stats[bin_idx]['mean'],
                    f'{target}_count': target_stats[bin_idx]['sum']
                })

        self.binning_config[feature][bin_id]['stats'] = bin_stats

    def transform(self, df, global_replace_na=None):
        """
        应用分箱规则转换数据

        参数:
            df: 需要转换的数据框
            global_replace_na: 全局缺失值替换值，若为None则使用每个分箱方案配置的replace_na
        """
        df_transformed = df.copy()

        for feature in self.features:
            if feature not in df.columns:
                raise ValueError(f"特征 {feature} 不在数据框中")

            if feature not in self.binning_config:
                raise ValueError(f"特征 {feature} 没有分箱配置，请先调用fit方法")

            # 应用该特征的所有分箱方案
            for bin_id, bin_config in self.binning_config[feature].items():
                # 确定缺失值替换值
                replace_na = bin_config['replace_na']
                if global_replace_na is not None:
                    replace_na = global_replace_na

                bins = bin_config['bins']
                bin_labels = list(range(len(bins) - 1))
                new_column = self._generate_feature_column_name(feature, bin_id)

                # 应用分箱
                df_transformed[new_column] = pd.cut(
                    df_transformed[feature],
                    bins=bins,
                    labels=bin_labels,
                    include_lowest=True
                )

                # 处理缺失值
                if df_transformed[feature].isna().any():
                    df_transformed[new_column] = df_transformed[new_column].cat.add_categories(replace_na)
                    df_transformed[new_column].fillna(replace_na, inplace=True)

                # 转换为整数类型
                df_transformed[new_column] = df_transformed[new_column].astype(int)

        return df_transformed

    def fit_transform(self, df, features_config, target=None, **kwargs):
        """训练分箱规则并转换数据"""
        self.fit(df, features_config, target,** kwargs)
        return self.transform(df)

    def save_config(self, file_path):
        """保存分箱配置到JSON文件"""
        # 创建目录（如果不存在）
        dir_path = os.path.dirname(file_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # 保存配置
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.binning_config, f, ensure_ascii=False, indent=2)

        print(f"分箱配置已保存到 {file_path}")

    def load_config(self, file_path):
        """从JSON文件加载分箱配置"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"配置文件 {file_path} 不存在")

        with open(file_path, 'r', encoding='utf-8') as f:
            self.binning_config = json.load(f)

        # 提取特征列表
        self.features = list(self.binning_config.keys())

        print(f"分箱配置已从 {file_path} 加载，共包含 {len(self.features)} 个特征的分箱方案")
        return self

    def get_feature_bin_strategies(self, feature):
        """获取指定特征的所有分箱方案"""
        if feature not in self.binning_config:
            return {}
        return self.binning_config[feature]

# 使用示例
if __name__ == "__main__":
    # 示例数据（包含缺失值）
    data = """user_id,age,monthly_income,browse_time_minutes,spending_frequency,gender,occupation,city,education,family_size,login_count_30d,member_level,browsed_categories,has_coupon,is_new_user,joined_promotion,has_notification_enabled,label
USER_000000,38.5,6438.46,,5.03,Male,Corporate Employee,San Antonio,Associate Degree,1,4,3,6,0,True,False,True,0
USER_000001,32.5,,22.99,6.18,Female,Teacher,Los Angeles,Master or Higher,2,6,1,3,0,False,False,False,0
USER_000002,,3196.08,3.83,1.74,Male,Civil Servant,San Antonio,Associate Degree,7,6,3,10,1,False,True,False,0
USER_000003,37.7,9286.22,8.99,8.06,Male,Corporate Employee,San Diego,Master or Higher,1,9,5,7,1,True,True,False,1
USER_000004,37.8,4658.54,12.12,,Male,Doctor,Other City,Master or Higher,4,5,3,4,1,False,True,True,1
USER_000005,18.0,5837.08,9.37,2.01,Male,Corporate Employee,New York,Associate Degree,8,7,4,6,1,False,True,False,1
USER_000006,46.1,4653.59,1.55,3.48,Female,Engineer,Other City,Bachelor Degree,2,6,3,4,0,False,False,False,0
USER_000007,23.1,6154.23,9.42,7.96,Female,Student,San Antonio,Bachelor Degree,4,8,3,4,1,False,False,False,1"""

    # 转换为DataFrame
    import io
    df = pd.read_csv(io.StringIO(data))

    # 定义每个特征的多种分箱配置
    feature_configs = {
        'age': [
            {'method': 'equal_width', 'n_bins': 3, 'replace_na': -1},
            {'method': 'kmeans', 'n_bins': 2, 'replace_na': -99}
        ],
        'monthly_income': [
            {'method': 'equal_freq', 'n_bins': 4, 'replace_na': -999},
            {'method': 'decision_tree', 'n_bins': 3}
        ],
        'browse_time_minutes': [
            {'method': 'kmeans', 'n_bins': 3}
        ]
    }

    # 初始化分箱器并训练
    binner = AdvancedFeatureBinner()
    binner.fit(df, feature_configs, target='label')

    # 保存配置
    binner.save_config('binning_configs/multi_strategy_config.json')

    # 转换数据
    df_transformed = binner.transform(df)

    # 查看转换结果
    print("转换后的分箱结果（仅展示分箱特征）:")
    binned_columns = [col for col in df_transformed.columns if any(col.startswith(f"{f}_") for f in feature_configs.keys())]
    print(df_transformed[binned_columns].head())

    # 查看age特征的所有分箱方案
    print("\nage特征的所有分箱方案:")
    import pprint
    pprint.pprint(binner.get_feature_bin_strategies('age'))

    # 演示加载配置
    print("\n从文件加载配置:")
    new_binner = AdvancedFeatureBinner()
    new_binner.load_config('binning_configs/multi_strategy_config.json')

    # 使用加载的配置进行转换
    df_new_transformed = new_binner.transform(df)
    print("\n使用加载的配置转换的结果（与之前对比）:")
    print(df_new_transformed[binned_columns].head())
