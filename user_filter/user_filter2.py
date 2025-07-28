import pandas as pd

def filter_dataframe_with_time(df):
    # 先按user_id和time排序，确保每组内时间从早到晚排列
    df_sorted = df.sort_values(by=['user_id', 'time'])

    # 按user_id分组处理
    def process_group(group):
        # 检查组内是否有label为1的记录
        if (group['label'] == 1).any():
            # 筛选出label为1的记录，取最后一条（时间最新的）
            return group[group['label'] == 1].iloc[-1]
        else:
            # 所有label都是0，取最后一条（时间最新的）
            return group.iloc[-1]

    # 应用分组处理函数
    result = df_sorted.groupby('user_id', group_keys=False).apply(process_group)
    return result

# 示例用法
if __name__ == "__main__":
    # 创建带时间字段的示例数据
    data = {
        'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
        'label': [0, 1, 0, 0, 0, 1, 1, 0, 0, 1],
        'time': pd.to_datetime([
            '2023-01-01', '2023-01-02', '2023-01-03',  # 用户1
            '2023-01-01', '2023-01-02',  # 用户2
            '2023-01-01', '2023-01-02', '2023-01-03',  # 用户3
            '2023-01-01', '2023-01-02'   # 用户4
        ]),
        'other_data': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    }
    df = pd.DataFrame(data)

    # 调用函数进行过滤
    filtered_df = filter_dataframe_with_time(df)
    print("原始数据:")
    print(df)
    print("\n过滤后的数据:")
    print(filtered_df)
