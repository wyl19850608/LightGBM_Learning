import pandas as pd

def extract_user_features(df, user_id_col='user_id', label_col='label'):
    """
    基于user_id和label提取特征，每个user_id只保留一条记录

    参数:
        df (pd.DataFrame): 包含用户数据的DataFrame
        user_id_col (str): 用户ID列名，默认为'user_id'
        label_col (str): 标签列名，默认为'label'

    返回:
        pd.DataFrame: 每个user_id保留一条记录的DataFrame
    """
    # 复制数据避免修改原DataFrame
    result_df = df.copy()

    # 确保label列是数值类型
    result_df[label_col] = pd.to_numeric(result_df[label_col], errors='coerce')

    # 按user_id分组处理
    def process_group(group):
        # 检查组中是否有label=1的记录
        has_positive = (group[label_col] == 1).any()

        if has_positive:
            # 如果有label=1的记录，取第一条label=1的记录
            return group[group[label_col] == 1].iloc[0:1]
        else:
            # 如果全是0或缺失值，取第一条记录
            return group.iloc[0:1]

    # 应用分组处理函数
    result = result_df.groupby(user_id_col, group_keys=False).apply(process_group)

    return result


# 使用示例
if __name__ == "__main__":
    # 设置pandas显示选项，确保全量打印
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # 示例数据
    data = {
        'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 5],
        'label': [0, 1, 0, 0, 0, 1, 0, 1, None, 0],
        'feature1': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'feature2': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    }
    df = pd.DataFrame(data)

    print("原始数据:")
    print(df)

    # 提取特征
    result = extract_user_features(df)

    print("\n处理后的数据:")
    print(result)
