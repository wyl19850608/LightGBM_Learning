import pandas as pd
import numpy as np
from typing import List, Optional, Literal

def handle_outliers_by_iqr(
        df: pd.DataFrame,
        features: List[str],
        fill_strategy: Literal["cap", "median"] = "cap",
        new_column_suffix: Optional[str] = "_handled",
        inplace: bool = False
) -> pd.DataFrame:
    """
    使用IQR法检测并处理连续值

    参数:
        df: 输入的DataFrame
        features: 需要处理的特征列表
        fill_strategy: 异常值填充策略
            - "cap": 截断（用上下界替换异常值）
            - "median": 用中位数替换异常值
        new_column_suffix: 新列的后缀，若为None则在原列修改
        inplace: 是否在原DataFrame上操作（仅当new_column_suffix为None时有效）

    返回:
        处理后的DataFrame
    """
    # 复制数据（如需创建新列或不原地修改）
    if not inplace or new_column_suffix is not None:
        df = df.copy()

    for feat in features:
        if feat not in df.columns:
            print(f"警告: 特征 {feat} 不在DataFrame中，已跳过")
            continue

        # 计算四分位和IQR
        q1 = df[feat].quantile(0.25)
        q3 = df[feat].quantile(0.75)
        iqr = q3 - q1

        # 计算上下界
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # 确定目标列（原列或新列）
        target_col = feat if new_column_suffix is None else f"{feat}{new_column_suffix}"

        # 检测异常值
        is_outlier_lower = df[feat] < lower_bound
        is_outlier_upper = df[feat] > upper_bound
        outlier_count = (is_outlier_lower | is_outlier_upper).sum()
        print(f"特征 {feat}: 检测到 {outlier_count} 个异常值")

        # 根据策略填充异常值
        if fill_strategy == "cap":
            # 截断策略：用上下界替换
            df[target_col] = df[feat].clip(lower=lower_bound, upper=upper_bound)
        elif fill_strategy == "median":
            # 中位数策略：用中位数替换
            median = df[feat].median()
            df.loc[is_outlier_lower | is_outlier_upper, target_col] = median
            # 正常数据直接复制
            if new_column_suffix is not None:
                df.loc[~(is_outlier_lower | is_outlier_upper), target_col] = df[feat]

    return df

# 示例用法
if __name__ == "__main__":
    # 创建含异常值的示例数据
    data = {
        "age": [18, 22, 25, 30, 35, 150, 40, 45, 50, 55, -5],  # 150和-5是异常值
        "income": [3000, 5000, 8000, 10000, 12000, 500000, 15000, 18000, 20000, 25000, 100]  # 500000和100是异常值
    }
    df = pd.DataFrame(data)
    print("原始数据:")
    print(df)

    # 1. 创建新列处理（默认截断策略）
    df1 = handle_outliers_by_iqr(
        df,
        features=["age", "income"],
        new_column_suffix="_iqr_capped"
    )
    print("\n1. 创建新列（截断处理）:")
    print(df1[["age", "age_iqr_capped", "income", "income_iqr_capped"]])

    # 2. 原地修改（中位数策略）
    df2 = handle_outliers_by_iqr(
        df,
        features=["age", "income"],
        fill_strategy="median",
        new_column_suffix=None,
        inplace=False  # 即使inplace=True，因复制过数据，原df不受影响
    )
    print("\n2. 原地修改（中位数处理）:")
    print(df2)
