import pandas as pd
import numpy as np
from typing import List, Optional, Literal, Tuple, Dict

def handle_outliers_by_iqr(
        df: pd.DataFrame,
        features: List[str],
        upper_strategy: Literal["cap", "keep"] = "cap",
        lower_strategy: Literal["cap", "keep"] = "cap",
        new_column_suffix: Optional[str] = "_handled",
        inplace: bool = False,
        iqr_multiplier: float = 1.5,
        lower_quantile: float = 0.25,
        upper_quantile: float = 0.75
) -> Tuple[pd.DataFrame, List[str], Dict[str, Dict[str, float]]]:
    """
     基于IQR（四分位距）方法检测并处理连续特征中的异常值，支持灵活配置处理策略。
     异常值判定标准：超出 [Q_low - iqr_multiplier×IQR, Q_high + iqr_multiplier×IQR] 范围的值，
     其中Q_low和Q_high为用户指定的分位数，IQR = Q_high - Q_low。

     参数:
         df: pd.DataFrame
             输入的数据集，包含需要处理的特征列。

         features: List[str]
             需要检测和处理异常值的特征名称列表。若列表中的特征不存在于df中，会跳过并警告。

         upper_strategy: Literal["cap", "keep"] = "cap"
             上界异常值的处理策略：
             - "cap": 用上界（Q_high + iqr_multiplier×IQR）替换超出上界的异常值
             - "keep": 保留上界异常值，不做处理

         lower_strategy: Literal["cap", "keep"] = "cap"
             下界异常值的处理策略：
             - "cap": 用下界（Q_low - iqr_multiplier×IQR）替换低于下界的异常值
             - "keep": 保留下界异常值，不做处理

         new_column_suffix: Optional[str] = "_handled"
             新增处理列的后缀。若为字符串（如"_iqr"），则在原特征列后添加新列存储处理结果；
             若为None，则直接在原列上修改（需配合inplace参数使用）。

         inplace: bool = False
             是否在原DataFrame上进行修改：
             - True: 直接修改输入的df，不创建副本（仅当new_column_suffix为None时有效）
             - False: 创建df的副本进行处理，不影响原数据

         iqr_multiplier: float = 1.5
             IQR的乘数，用于计算异常值判定的上下界。常用值：
             - 1.5: 标准IQR法，检测常规异常值
             - 3.0: 更严格，检测极端异常值

         lower_quantile: float = 0.25
             下分位数比例（范围：0 < lower_quantile < upper_quantile < 1），默认0.25（第25百分位数）。
             用于计算Q_low（下界分位数）。

         upper_quantile: float = 0.75
             上分位数比例（范围：0 < lower_quantile < upper_quantile < 1），默认0.75（第75百分位数）。
             用于计算Q_high（上界分位数）。

     返回:
         Tuple[pd.DataFrame, List[str], Dict[str, Dict[str, float]]]
             - 第一个元素：处理后的DataFrame（若inplace=True，则为原df的引用）
             - 第二个元素：所有被处理的列名列表（新增列或被修改的原列，去重且保持顺序）
             - 第三个元素：填充值字典，结构为{特征列名: {"lower_bound": 下界值, "upper_bound": 上界值}}
     """
    # 验证分位数参数有效性
    if not (0 < lower_quantile < upper_quantile < 1):
        raise ValueError("分位数参数必须满足 0 < lower_quantile < upper_quantile < 1")

    # 控制数据复制，优化内存
    if not inplace or new_column_suffix is not None:
        df = df.copy()

    # 记录所有被处理的列名
    processed_columns = []
    # 记录填充值（上下界）
    fill_values = {}

    for feat in features:
        if feat not in df.columns:
            print(f"警告: 特征 {feat} 不存在，已跳过")
            continue

        # 计算分位数和边界
        q_low = df[feat].quantile(lower_quantile)
        q_high = df[feat].quantile(upper_quantile)
        iqr = q_high - q_low
        lower_bound = q_low - iqr_multiplier * iqr
        upper_bound = q_high + iqr_multiplier * iqr

        # 存储当前特征的填充值
        fill_values[feat] = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }

        # 确定目标列
        target_col = feat if new_column_suffix is None else f"{feat}{new_column_suffix}"
        if new_column_suffix is not None and target_col not in df.columns:
            df[target_col] = df[feat].copy()

        # 检测异常值
        is_outlier_lower = df[feat] < lower_bound
        is_outlier_upper = df[feat] > upper_bound
        lower_count = is_outlier_lower.sum()
        upper_count = is_outlier_upper.sum()

        # 处理异常值（仅当有异常值且策略为cap时）
        has_processing = False
        if lower_strategy == "cap" and lower_count > 0:
            df.loc[is_outlier_lower, target_col] = lower_bound
            has_processing = True
        if upper_strategy == "cap" and upper_count > 0:
            df.loc[is_outlier_upper, target_col] = upper_bound
            has_processing = True

        # 记录被处理的列（即使没有异常值，只要是目标列也记录）
        processed_columns.append(target_col)  # 修正了此处的变量名拼写错误
        print(f"特征 {feat}: 处理列={target_col}, 下界异常值{lower_count}个, 上界异常值{upper_count}个")

    # 去重并保持顺序（避免重复处理同一列的情况）
    processed_columns = list(dict.fromkeys(processed_columns))
    return df, processed_columns, fill_values


# 示例用法
if __name__ == "__main__":
    data = {
        "values": [500, 3, 5, 7, 9, 11, 13, 15, 17, 19, 100, -110],
        "score": [800, 85, 90, 95, 100, 105, 110, 115, 120, 125, 200, -300]
    }
    df = pd.DataFrame(data)
    print("原始数据:")
    print(df)

    # 处理异常值
    processed_df, handled_cols, fill_values = handle_outliers_by_iqr(
        df,
        features=["values", "score"],
        new_column_suffix="_iqr"
    )

    print("\n处理后的DataFrame列:", processed_df.columns.tolist())
    print("被处理的列名:", handled_cols)
    print("填充值（上下界）:")
    for feat, bounds in fill_values.items():
        print(f"  {feat}: 下界={bounds['lower_bound']}, 上界={bounds['upper_bound']}")
    print("\n处理结果预览:")
    print(processed_df)
