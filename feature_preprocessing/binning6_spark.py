from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (col, when, lit, count, mean, min, max,
                                   sum as spark_sum, isnan)
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType, BooleanType
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.functions import vector_to_array
import pandas as pd
import numpy as np
import json
import os
from scipy.stats import chi2_contingency

class SparkFeatureBinner:
    """基于Spark DataFrame的高级特征分箱处理器（兼容所有Spark版本）"""

    def __init__(self, spark: SparkSession = None):
        """初始化分箱器"""
        self.spark = spark or SparkSession.builder \
            .appName("FeatureBinner") \
            .getOrCreate()

        self.binning_config = {}  # 存储分箱配置
        self.features = []        # 所有分箱特征列表

    def _generate_bin_id(self, method, n_bins):
        """生成分箱方案唯一标识"""
        return f"{method}_{n_bins}"

    def _generate_feature_column_name(self, feature, bin_id):
        """生成新特征列名，格式: 原始特征_method_n_bins"""
        return f"{feature}_{bin_id}"

    def _spark_cut(self, df: DataFrame, feature: str, bins: list, labels: list, include_lowest: bool = True):
        """自定义义Spark分箱函数"""
        bins = sorted(bins)
        num_bins = len(bins) - 1

        if len(labels) != num_bins:
            raise ValueError(f"分箱标签数量({len(labels)})与分箱数量({num_bins})不匹配")

        # 初始化分箱条件（同时处理Null和NaN）
        bin_condition = when(col(feature).isNull(), None) \
            .when(isnan(col(feature)), None)

        # 处理第一个区间
        if include_lowest:
            bin_condition = bin_condition.when(
                col(feature) <= bins[1],
                lit(labels[0])
            )
        else:
            bin_condition = bin_condition.when(
                (col(feature) > bins[0]) & (col(feature) <= bins[1]),
                lit(labels[0])
            )

        # 处理中间区间
        for i in range(1, num_bins - 1):
            bin_condition = bin_condition.when(
                (col(feature) > bins[i]) & (col(feature) <= bins[i+1]),
                lit(labels[i])
            )

        # 处理最后一个区间
        bin_condition = bin_condition.when(
            col(feature) > bins[-2],
            lit(labels[-1])
        )

        return df.withColumn(f"{feature}_temp_bin", bin_condition)

    def fit(self, df: DataFrame, features_config: dict, target: str = None, min_samples_leaf: float = 0.05, random_state: int = 42):
        """训练分箱规则（兼容所有Spark版本）"""
        if not isinstance(features_config, dict):
            raise ValueError("features_config参数必须是字典类型")

        self.features = list(features_config.keys())
        self.binning_config = {feature: {} for feature in self.features}

        # 计算总样本数
        total_samples = df.count()
        # 计算最小样本数（不依赖math模块）
        calculated_samples = int(total_samples * min_samples_leaf)
        min_samples = 1 if calculated_samples < 1 else calculated_samples

        for feature, bin_strategies in features_config.items():
            if feature not in df.columns:
                raise ValueError(f"特征 {feature} 不在DataFrame中")

            if not isinstance(bin_strategies, list) or len(bin_strategies) == 0:
                raise ValueError(f"特征 {feature} 的分箱方案必须是非空列表")

            # 统计并过滤NaN和Null值
            null_count = df.filter(col(feature).isNull() | isnan(col(feature))).count()
            if null_count > 0:
                print(f"警告：特征 {feature} 包含 {null_count} 个缺失值（NaN/Null），将全部过滤")

            # 严格过滤：移除所有NaN和Null的行
            non_null_df = df.filter(col(feature).isNotNull() & ~isnan(col(feature)))
            if non_null_df.count() == 0:
                print(f"警告：特征 {feature} 过滤后无有效数据，跳过分箱处理")
                continue

            # 获取特征的统计信息（基于过滤后的数据）
            feature_stats = non_null_df.select(
                min(feature).alias("min_val"),
                max(feature).alias("max_val")
            ).collect()[0]

            min_val = feature_stats["min_val"]
            max_val = feature_stats["max_val"]

            # 为每个分箱方案训练
            for strategy in bin_strategies:
                if not isinstance(strategy, dict) or 'method' not in strategy or 'n_bins' not in strategy:
                    raise ValueError(f"特征 {feature} 的分箱方案配置不正确")

                method = strategy['method']
                n_bins = strategy['n_bins']
                replace_na = strategy.get('replace_na', -1)
                bin_id = self._generate_bin_id(method, n_bins)

                self.binning_config[feature][bin_id] = {
                    'method': method,
                    'n_bins': n_bins,
                    'replace_na': replace_na
                }

                # 等宽分箱
                if method == 'equal_width':
                    bins = np.linspace(min_val, max_val, n_bins + 1).tolist()
                    self.binning_config[feature][bin_id]['bins'] = bins

                # 等频分箱
                elif method == 'equal_freq':
                    quantiles = [i / n_bins for i in range(n_bins + 1)]
                    # 使用过滤后的非空数据计算分位数
                    percentiles = non_null_df.approxQuantile(feature, quantiles, 0.01)
                    bins = sorted(list(set(percentiles)))
                    if len(bins) < n_bins + 1:
                        bins = np.linspace(min_val, max_val, n_bins + 1).tolist()
                    self.binning_config[feature][bin_id]['bins'] = bins

                # K-means聚类分箱
                elif method == 'kmeans':
                    # 创建特征向量，明确处理无效值
                    assembler = VectorAssembler(
                        inputCols=[feature],
                        outputCol="features",
                        handleInvalid="skip"  # 跳过含无效值的行
                    )
                    data_with_features = assembler.transform(non_null_df)

                    # 将向量转换为数组以安全访问元素
                    data_with_array = data_with_features.withColumn(
                        "features_array",
                        vector_to_array(col("features"))
                    )

                    # 过滤特征向量中的无效值
                    clean_data = data_with_array.filter(
                        col("features").isNotNull() &
                        col("features_array").isNotNull() &
                        ~isnan(col("features_array")[0])
                    ).drop("features_array")

                    # 确保有足够数据进行聚类
                    if clean_data.count() < n_bins:
                        print(f"警告：特征 {feature} 有效数据不足（{clean_data.count()}条），无法进行K-means聚类（需要至少{n_bins}条），自动降级为等宽分箱")
                        bins = np.linspace(min_val, max_val, n_bins + 1).tolist()
                        self.binning_config[feature][bin_id]['bins'] = bins
                        self.binning_config[feature][bin_id]['method'] = 'equal_width_fallback'
                        continue

                    # 训练K-means模型
                    kmeans = KMeans(
                        featuresCol="features",
                        predictionCol="cluster",
                        k=n_bins,
                        seed=random_state
                    )
                    model = kmeans.fit(clean_data)

                    # 生成分箱边界
                    centers = sorted([float(center[0]) for center in model.clusterCenters()])
                    bins = [min_val - 1e-10]
                    for i in range(len(centers) - 1):
                        bins.append((centers[i] + centers[i+1]) / 2)
                    bins.append(max_val + 1e-10)
                    self.binning_config[feature][bin_id]['bins'] = bins

                # 决策树分箱（兼容所有Spark版本的实现）
                elif method == 'decision_tree':
                    if target is None:
                        raise ValueError(f"特征 {feature} 使用决策树分箱需要指定目标变量target")

                    # 严格处理特征向量
                    assembler = VectorAssembler(
                        inputCols=[feature],
                        outputCol="features",
                        handleInvalid="skip"
                    )
                    data_with_features = assembler.transform(non_null_df)

                    # 处理向量访问
                    data_with_array = data_with_features.withColumn(
                        "features_array",
                        vector_to_array(col("features"))
                    )

                    # 过滤目标变量也不为空的数据
                    clean_data = data_with_array.filter(
                        col(target).isNotNull() & ~isnan(col(target)) &
                        col("features").isNotNull() &
                        col("features_array").isNotNull() &
                        ~isnan(col("features_array")[0])
                    ).drop("features_array")

                    if clean_data.count() == 0:
                        print(f"警告：特征 {feature} 目标变量含缺失值，决策树分箱降级为等宽分箱")
                        bins = np.linspace(min_val, max_val, n_bins + 1).tolist()
                        self.binning_config[feature][bin_id]['bins'] = bins
                        continue

                    dt = DecisionTreeClassifier(
                        featuresCol="features",
                        labelCol=target,
                        maxBins=n_bins,
                        minInstancesPerNode=min_samples,
                        seed=random_state
                    )
                    model = dt.fit(clean_data)

                    # 兼容所有Spark版本的阈值提取方法
                    thresholds = []
                    try:
                        # 方法1: 尝试新版本API
                        nodes = model.treeNodeCollector().collect()
                        for node in nodes:
                            if node.split:
                                thresholds.append(float(node.split.threshold))
                    except AttributeError:
                        try:
                            # 方法2: 尝试旧版本API - nodeAt
                            java_model = model._java_obj
                            node_count = java_model.numNodes()
                            for i in range(node_count):
                                node = java_model.nodeAt(i)
                                if not node.isLeaf():
                                    thresholdsolds.append(float(node.split().threshold()))
                        except Exception:
                            try:
                                # 方法3: 尝试另一种旧版本API - 树结构访问
                                java_model = model._java_obj
                                tree = java_model.rootNode()
                                # 使用递归函数提取阈值
                                def extract_thresholds(node, thresholds):
                                    if not node.isLeaf():
                                        thresholdsolds.append(float(node.split().threshold()))
                                        extract_thresholds(node.leftChild(), thresholds)
                                        extract_thresholds(node.rightChild(), thresholds)
                                extract_thresholds(tree, thresholds)
                            except Exception as e:
                                # 所有方法都失败时降级为等宽分箱
                                print(f"警告：无法提取决策树阈值，可能是Spark版本不兼容，降级为等宽分箱。错误: {str(e)}")
                                bins = np.linspace(min_val, max_val, n_bins + 1).tolist()
                                self.binning_config[feature][bin_id]['bins'] = bins
                                self.binning_config[feature][bin_id]['method'] = 'equal_width_fallback'
                                continue

                    thresholds = sorted(list(set(thresholds)))
                    # 确保分箱数量合理
                    if len(thresholds) == 0:
                        bins = np.linspace(min_val, max_val, n_bins + 1).tolist()
                    else:
                        bins = [min_val - 1e-10] + thresholds + [max_val + 1e-10]
                        # 确保分箱数量不超过指定数量
                        if len(bins) - 1 > n_bins:
                            bins = np.linspace(min_val, max_val, n_bins + 1).tolist()

                    self.binning_config[feature][bin_id]['bins'] = bins

                # 卡方分箱
                elif method == 'chi2':
                    if target is None:
                        raise ValueError(f"特征 {feature} 使用卡方分箱需要指定目标变量target")

                    # 过滤目标变量非空的数据
                    clean_data = non_null_df.filter(col(target).isNotNull() & ~isnan(col(target)))
                    if clean_data.count() == 0:
                        print(f"警告：特征 {feature} 目标变量含缺失值，卡方分箱降级为等宽分箱")
                        bins = np.linspace(min_val, max_val, n_bins + 1).tolist()
                        self.binning_config[feature][bin_id]['bins'] = bins
                        continue

                    unique_values = [row[feature] for row in clean_data.select(feature).distinctinct().orderBy(feature).collect()]

                    if len(unique_values) <= n_bins:
                        bins = [min_val - 1e-10] + unique_values + [max_val + 1e-10]
                        self.binning_config[feature][bin_id]['bins'] = bins
                        continue

                    bins = unique_values.copy()
                    while len(bins) > n_bins:
                        min_chi = float('inf')
                        merge_idx = 0

                        for i in range(len(bins) - 1):
                            lower1, upper1 = bins[i], bins[i+1]
                            lower2, upper2 = bins[i+1], bins[i+2] if i+2 < len(bins) else max_val + 1e-10

                            bin1 = clean_data.filter((col(feature) >= lower1) & (col(feature) < upper1))
                            bin1_counts = bin1.groupBy(target).count().collect()
                            bin1_0 = next((row['count'] for row in bin1_counts if row[target] == 0), 0)
                            bin1_1 = next((row['count'] for row in bin1_counts if row[target] == 1), 0)

                            bin2 = clean_data.filter((col(feature) >= lower2) & (col(feature) < upper2))
                            bin2_counts = bin2.groupBy(target).count().collect()
                            bin2_0 = next((row['count'] for row in bin2_counts if row[target] == 0), 0)
                            bin2_1 = next((row['count'] for row in bin2_counts if row[target] == 1), 0)

                            observed = np.array([[bin1_0, bin1_1], [bin2_0, bin2_1]])

                            if observed.sum() == 0:
                                chi2 = 0
                            else:
                                chi2, _, _, _ = chi2_contingency(observed)

                            if chi2 < min_chi:
                                min_chi = chi2
                                merge_idx = i

                        bins.pop(merge_idx + 1)

                    bins = [min_val - 1e-10] + bins + [max_val + 1e-10]
                    self.binning_config[feature][bin_id]['bins'] = bins

                else:
                    raise ValueError(f"不支持的分箱方法: {method}")

                # 计算每个分箱的统计信息
                self._calculate_bin_stats(df, feature, bin_id, target)

        return self

    def _calculate_bin_stats(self, df: DataFrame, feature: str, bin_id: str, target: str = None):
        """计算每个分箱方案的统计信息"""
        bins = self.binning_config[feature][bin_id]['bins']
        bin_labels = list(range(len(bins) - 1))

        # 应用分箱
        binned_df = self._spark_cut(df, feature, bins, bin_labels)

        # 计算基本统计信息
        stats_df = binned_df.groupBy(f"{feature}_temp_bin").agg(
            count(feature).alias("count"),
            mean(feature).alias("mean"),
            min(feature).alias("min"),
            max(feature).alias("max")
        ).orderBy(f"{feature}_temp_bin")

        # 转换为字典
        bin_stats = {}
        for row in stats_df.collect():
            bin_idx = row[f"{feature}_temp_bin"]
            if bin_idx is not None:
                bin_stats[int(bin_idx)] = {
                    "count": row["count"],
                    "mean": float(row["mean"]) if row["mean"] is not None else None,
                    "min": float(row["min"]) if row["min"] is not None else None,
                    "max": float(row["max"]) if row["max"] is not None else None
                }

        # 如果有目标变量，计算目标变量相关统计
        if target and target in df.columns:
            target_stats_df = binned_df.groupBy(f"{feature}_temp_bin").agg(
                mean(target).alias(f"{target}_mean"),
                spark_sum(target).alias(f"{target}_count")
            ).orderBy(f"{feature}_temp_bin")

            for row in target_stats_df.collect():
                bin_idx = row[f"{feature}_temp_bin"]
                if bin_idx is not None and bin_idx in bin_stats:
                    bin_stats[int(bin_idx)].update({
                        f"{target}_mean": float(row[f"{target}_mean"]) if row[f"{target}_mean"] is not None else None,
                        f"{target}_count": int(row[f"{target}_count"]) if row[f"{target}_count"] is not None else None
                    })

        self.binning_config[feature][bin_id]['stats'] = bin_stats

        # 移除临时列
        return binned_df.drop(f"{feature}_temp_bin")

    def transform(self, df: DataFrame, global_replace_na: int = None) -> DataFrame:
        """应用分箱规则转换数据"""
        df_transformed = df

        for feature in self.features:
            if feature not in df.columns:
                raise ValueError(f"特征 {feature} 不在DataFrame中")

            if feature not in self.binning_config:
                continue  # 跳过未分箱的特征

            # 应用该特征的所有分箱方案
            for bin_id, bin_config in self.binning_config[feature].items():
                replace_na = bin_config['replace_na']
                if global_replace_na is not None:
                    replace_na = global_replace_na

                bins = bin_config['bins']
                bin_labels = list(range(len(bins) - 1))
                new_column = self._generate_feature_column_name(feature, bin_id)

                # 应用分箱
                df_transformed = self._spark_cut(df_transformed, feature, bins, bin_labels)

                # 处理缺失值并转换为整数类型
                df_transformed = df_transformed.withColumn(
                    new_column,
                    when(
                        col(f"{feature}_temp_bin").isNull() |
                        isnan(col(f"{feature}_temp_bin")),
                        lit(replace_na)
                    ).otherwise(col(f"{feature}_temp_bin").cast(IntegerType()))
                ).drop(f"{feature}_temp_bin")

        return df_transformed

    def fit_transform(self, df: DataFrame, features_config: dict, target: str = None, **kwargs) -> DataFrame:
        """训练分箱规则并转换数据"""
        self.fit(df, features_config, target,** kwargs)
        return self.transform(df)

    def save_config(self, file_path: str):
        """保存分箱配置到JSON文件"""
        dir_path = os.path.dirname(file_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.binning_config, f, ensure_ascii=False, indent=2)

        print(f"分箱配置已保存到 {file_path}")

    def load_config(self, file_path: str):
        """从JSON文件加载分箱配置"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"配置文件 {file_path} 不存在")

        with open(file_path, 'r', encoding='utf-8') as f:
            self.binning_config = json.load(f)

        self.features = list(self.binning_config.keys())
        print(f"分箱配置已从 {file_path} 加载，共包含 {len(self.features)} 个特征的分箱方案")
        return self

    def get_feature_bin_strategies(self, feature: str) -> dict:
        """获取指定特征的所有分箱方案"""
        if feature not in self.binning_config:
            return {}
        return self.binning_config[feature]

# 辅助函数：创建兼容的Spark DataFrame
def create_spark_df_from_pandas(spark, pandas_df):
    """从pandas DataFrame创建Spark DataFrame，兼容高版本pandas"""
    # 定义schema
    schema = StructType()
    for column, dtype in pandas_df.dtypes.items():
        # 映射pandas类型到Spark类型
        if pd.api.types.is_integer_dtype(dtype):
            spark_type = IntegerType()
        elif pd.api.types.is_float_dtype(dtype):
            spark_type = DoubleType()
        elif pd.api.types.is_bool_dtype(dtype):
            spark_type = BooleanType()
        else:
            spark_type = StringType()

        schema.add(StructField(column, spark_type, nullable=True))

    # 转换数据为列表字典格式，替换NaN为None
    data = pandas_df.where(pd.notnull(pandas_df), None).to_dict('records')

    # 创建Spark DataFrame
    return spark.createDataFrame(data, schema=schema)

# 使用示例
if __name__ == "__main__":
    # 初始化SparkSession
    spark = SparkSession.builder \
        .appName("FeatureBinningExample") \
        .master("local[*]") \
        .getOrCreate()

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

    # 转换为Spark DataFrame（替换NaN为None）
    from io import StringIO
    pandas_df = pd.read_csv(StringIO(data))
    df = create_spark_df_from_pandas(spark, pandas_df)

    # 定义分箱配置
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
    binner = SparkFeatureBinner(spark)
    binner.fit(df, feature_configs, target='label')

    # 保存配置
    binner.save_config('binning_configs/multi_strategy_config.json')

    # 转换数据
    df_transformed = binner.transform(df)

    # 查看结果
    print("转换后的分箱结果:")
    binned_columns = [col for col in df_transformed.columns if any(col.startswith(f"{f}_") for f in feature_configs.keys())]
    df_transformed.select(binned_columns).show()

    # 加载配置并重新转换
    new_binner = SparkFeatureBinner(spark)
    new_binner.load_config('binning_configs/multi_strategy_config.json')
    df_new_transformed = new_binner.transform(df)

    print("\n使用加载的配置转换的结果:")
    df_new_transformed.select(binned_columns).show()

    spark.stop()
