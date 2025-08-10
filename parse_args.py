import argparse

def parse_args():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='LightGBM模型预测脚本')

    # 添加需要解析的参数
    parser.add_argument('-output_path', required=True,
                        help='输出结果的路径')
    parser.add_argument('-model_path', required=True,
                        help='模型文件的路径')
    parser.add_argument('-labelencoder_path', required=True,
                        help='标签编码器文件的路径')
    parser.add_argument('input_file',  #  positional参数（无-前缀），对应脚本中的{}
                        help='输入数据文件路径')

    # 解析参数并返回
    return parser.parse_args()

if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()

    # 打印参数（测试用）
    print(f"输出路径: {args.output_path}")
    print(f"模型路径: {args.model_path}")
    print(f"标签编码器路径: {args.labelencoder_path}")
    print(f"输入文件: {args.input_file}")

    # 这里可以添加实际的预测逻辑
    # ...
