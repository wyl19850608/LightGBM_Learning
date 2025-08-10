import pandas as pd
from typing import List  # 从typing模块导入List

def keep_specific_columns_inplace(df: pd.DataFrame, columns_to_keep: List[str]) -> None:
    """
    原地操作：仅保留DataFrame中指定的列，删除其他所有列（不创建副本）

    参数:
        df: 输入的DataFrame（将被直接修改）
        columns_to_keep: 需要保留的列名列表
    """
    # 检查需要保留的列是否存在
    missing_columns = [col for col in columns_to_keep if col not in df.columns]
    if missing_columns:
        print(f"警告：以下列不存在于DataFrame中，已忽略: {missing_columns}")

    # 计算需要删除的列（所有不在保留列表中的列）
    columns_to_drop = [col for col in df.columns if col not in columns_to_keep]

    if columns_to_drop:
        # 原地删除多余的列（不产生副本）
        df.drop(columns=columns_to_drop, inplace=True)
        print(f"已保留列: {df.columns.tolist()}")
    else:
        print("所有列均已在保留列表中，无需修改")

# 示例用法
if __name__ == "__main__":
    # 创建示例DataFrame
    data = {
        "id": [1, 2, 3],
        "tags": [["a", "b"], ["c"], ["d", "e", "f"]],
        "name": ["Alice", "Bob", "Charlie"],
        "scores": [[90, 85], [78], [92, 88]],
        "info": [{"key": "value"}, None, {}]
    }
    df = pd.DataFrame(data)
    print("原始列:", df.columns.tolist())

    # 原地保留id和info列（原DataFrame将被直接修改）
    keep_specific_columns_inplace(df, ["id", "info"])
    print("处理后列:", df.columns.tolist())
    print("\n处理后的数据:")
    print(df)
