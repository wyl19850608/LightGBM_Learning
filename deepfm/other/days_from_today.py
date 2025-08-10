import pandas as pd

def add_days_from_today(df, date_column, new_column=None, handle_nulls=True):
    """
    给DataFrame添加一列，计算日期列中每个日期距离当天的天数，可自动生成新列名并全量打印

    参数:
        df (pd.DataFrame): 包含日期列的DataFrame
        date_column (str): 日期列的列名
        new_column (str): 新列的列名，默认为None，此时将自动生成为"{date_column}_days_from_today"
        handle_nulls (bool): 是否用当前时间处理空值，默认为True

    返回:
        pd.DataFrame: 包含新列的DataFrame
    """
    # 复制DataFrame避免修改原数据
    result_df = df.copy()

    # 获取当前日期（不含时间部分）
    today = pd.Timestamp.today().normalize()

    # 自动生成新列名
    if new_column is None:
        new_column = f"{date_column}_days_from_today"

    # 将日期列转换为datetime类型
    result_df[date_column] = pd.to_datetime(result_df[date_column], errors='coerce')

    # 检查是否有转换失败的日期
    invalid_dates = result_df[date_column].isna().sum()
    if invalid_dates > 0:
        print(f"警告：有{invalid_dates}个日期值无法转换为有效的日期格式")

        if handle_nulls:
            print(f"提示：将使用当前时间（{today.date()}）处理空值")
            # 用当前时间填充空值
            result_df[date_column] = result_df[date_column].fillna(today)

    # 计算天数差
    result_df[new_column] = (today - result_df[date_column]).dt.days

    return result_df


# 使用示例
if __name__ == "__main__":
    # 设置pandas显示选项，确保全量打印不省略
    pd.set_option('display.max_columns', None)  # 显示所有列
    pd.set_option('display.max_rows', None)     # 显示所有行
    pd.set_option('display.width', None)        # 自动调整宽度
    pd.set_option('display.max_colwidth', None) # 显示列的完整内容

    # 示例数据（包含更多行以便测试）
    data = {
        'create_time': [
            '2024-06-23', None, 'invalid_date', '2024-06-15',
            '2024-01-01', '2024-12-31', '2023-06-23', '2025-06-23'
        ],
        'expire_time': [
            '2024-07-01', '', '2024-06-30', '2024-08-01',
            '2024-06-01', '2024-09-15', '2024-06-23', '2024-05-01'
        ]
    }
    df = pd.DataFrame(data)

    # 处理日期列
    df = add_days_from_today(df, 'create_time')
    df = add_days_from_today(df, 'expire_time')

    # 全量打印DataFrame
    print("处理后的完整数据:")
    print(df)

    # 如需恢复默认设置，可以使用：
    # pd.reset_option('all')
