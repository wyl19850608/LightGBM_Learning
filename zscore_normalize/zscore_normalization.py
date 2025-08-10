import pandas as pd
import numpy as np
from typing import List, Optional,Tuple

def zscore_normalize(df: pd.DataFrame, features: List[str], inplace: bool = False,
                     stats: Optional[dict] = None) -> (pd.DataFrame, dict):
    """
    对DataFrame中的连续特征进行Z-Score归一化处理
    Z-Score公式: z = (x - μ) / σ，其中μ为均值，σ为标准差

    参数:
        df: 包含待处理特征的DataFrame
        features: 需要归一化的连续特征列表
        inplace: 是否在原DataFrame上进行操作
        stats: 可选，已有的均值和标准差字典（格式: {feature: (mean, std)}）
               用于测试集归一化（复用训练集的统计量）

    返回:
        处理后的DataFrame和包含均值、标准差的字典
    """
    # 复制数据（如果不原地操作）
    if not inplace:
        df = df.copy()

    # 存储每个特征的均值和标准差
    zscore_stats = {}

    for feat in features:
        if feat not in df.columns:
            print(f"警告: 特征 {feat} 不在DataFrame中，已跳过")
            continue

        # 对于测试集，使用训练集的统计量
        if stats is not None and feat in stats:
            mean, std = stats[feat]
        else:
            # 计算当前特征的均值和标准差（跳过NaN值）
            mean = df[feat].mean(skipna=True)
            std = df[feat].std(skipna=True)

            # 处理标准差为0的情况（避免除零错误）
            if std < 1e-10:
                std = 1e-10
                print(f"警告: 特征 {feat} 的标准差为0，已替换为{std}")

            zscore_stats[feat] = (mean, std)

        # 执行Z-Score归一化（原地修改）
        df[feat] = (df[feat] - mean) / std

    # 如果是训练集且未提供stats，返回计算的统计量；否则返回输入的stats
    return_df = df if inplace else df
    return_stats = zscore_stats if stats is None else stats

    return return_df, return_stats



# 示例用法
if __name__ == "__main__":
    # 定义连续特征列表（模拟训练时使用的特征）
    continuous_features = ['age', 'price', 'score']

    # 模拟训练阶段得到的统计量（实际应用中应从训练过程中保存和加载）
    stats = {
        'age': (30.0, 5.0),      # (均值, 标准差)
        'price': (200.0, 30.0),  # (均值, 标准差)
        'score': (85.0, 10.0)    # (均值, 标准差)
    }

    # 模拟新的预测数据
    prediction_data = {
        'age': [28, None, 37],
        'price': [190, 260, 230],
        'score': [86, 92, 79],
        'new_feature': [1, 2, 3]  # 新增未在训练中出现的特征
    }
    pred_df = pd.DataFrame(prediction_data)
    print("\n原始预测数据:")
    print(pred_df)



    # 使用训练阶段得到的统计量进行预处理
    test_df_normalized, _ = zscore_normalize(pred_df, continuous_features,
                                             inplace=False, stats=stats)
    print("\n测试数据归一化后 (复用训练集统计量):")
    print(test_df_normalized)



