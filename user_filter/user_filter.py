import pandas as pd

def filter_dataframe(df):
    # 按user_id分组处理
    def process_group(group):
        # 检查组内是否有label为1的记录
        if (group['label'] == 1).any():
            # 筛选出label为1的记录，取最后一条
            return group[group['label'] == 1].iloc[-1]
        else:
            # 所有label都是0，取最后一条记录
            return group.iloc[-1]

    # 应用分组处理函数
    result = df.groupby('user_id', group_keys=False).apply(process_group)
    return result

# 示例用法
if __name__ == "__main__":
    # 创建示例数据
    data = {
        'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
        'label': [0, 1, 0, 0, 0, 1, 1, 0, 0, 1],
        'other_data': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    }
    df = pd.DataFrame(data)

    # 调用函数进行过滤
    filtered_df = filter_dataframe(df)
    print("原始数据:")
    print(df)
    print("\n过滤后的数据:")
    print(filtered_df)