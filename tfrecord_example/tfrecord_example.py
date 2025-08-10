import tensorflow as tf
import glob
import numpy as np

# 全局变量定义
all_features = ["user_id", "age", "gender", "click_count", "purchase_count", "label", "history_items"]
all_feature_types = ["int", "int", "int", "int", "int", "int", "ARRAY<int>"]
model_features = ["user_id", "age", "gender", "click_count", "purchase_count", "history_items"]

def create_sample_data(num_samples=1000):
    """生成示例数据，包含重复用户ID和正负样本"""
    data = []
    for i in range(num_samples):
        # 控制部分用户只有负样本
        has_positive = (i % 5 != 0)  # 每5个用户有1个全是负样本
        label = np.random.randint(0, 2) if has_positive else 0

        sample = {
            "user_id": str(i % 100),  # 限制用户ID范围（0-99），确保有重复用户
            "age": np.random.randint(18, 70),
            "gender": np.random.randint(0, 2),
            "click_count": np.random.randint(0, 100),
            "purchase_count": np.random.randint(0, 20),
            "label": label,
            "history_items": np.random.randint(1000, 10000, size=10)  # 长度为10的历史物品列表
        }
        data.append(sample)
    return data

def write_tfrecord(data, output_path):
    """将数据写入TFRecord文件"""
    writer = tf.io.TFRecordWriter(output_path)
    for sample in data:
        feature = {}
        # 处理标量特征
        for feature_name in ["user_id", "age", "gender", "click_count", "purchase_count", "label"]:
            feature[feature_name] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[sample[feature_name]])
            )
        # 处理数组特征（历史物品）
        feature["history_items"] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=sample["history_items"])
        )
        # 构建Example并写入
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()
    print(f"TFRecord文件已写入: {output_path}")

def read_tfrecord(example):
    """解析TFRecord文件中的样本"""
    format_builder = {}
    filter_feature_names = []

    # 构建解析格式
    for feature in all_features:
        i = all_features.index(feature)
        f_type = all_feature_types[i]

        if f_type in ["int", "bigint"]:
            format_builder[feature] = tf.io.FixedLenFeature([], tf.int64)
        elif f_type == "float":
            format_builder[feature] = tf.io.FixedLenFeature([], tf.float32)
        elif f_type == "string":
            format_builder[feature] = tf.io.FixedLenFeature([], tf.string)
        else:  # 数组类型
            format_builder[feature] = tf.io.FixedLenSequenceFeature(
                [], tf.int64, allow_missing=True
            )
            filter_feature_names.append(feature)

    # 解析样本
    example = tf.io.parse_single_example(example, format_builder)
    labels = example["label"]

    # 过滤不需要的特征
    for feature_name in all_features:
        if (feature_name not in model_features
                and feature_name not in filter_feature_names
                and feature_name != "request_id"):
            example.pop(feature_name, None)

    # 转换int64为int32（节省内存）
    for key in example:
        if example[key].dtype == tf.int64:
            example[key] = tf.cast(example[key], tf.int32)

    return example, labels

def group_by_user_preprocessing(dataset):
    """
    按user_id分组，每组内优先选择第一条label=1的样本，若无则选择第一条样本
    """

    def key_func(x, y):
        # 分组键需要显式转换为int64（解决类型不匹配问题）
        if tf.equal(y, -1):
            return tf.constant(-1, dtype=tf.int64)

        # 关键1：双哈希+1万亿桶降低冲突
        return tf.strings.to_hash_bucket_strong(
            x["user_id"],
            num_buckets=10**12,  # 1万亿桶，冲突概率≈1e-8（1亿用户）
            seed=[123456, 789012]  # 固定随机种子，确保相同user_id映射一致
        )

    def reduce_func(key, window_dataset):
        """处理每个用户的样本窗口"""
        # 过滤正样本
        positive_samples = window_dataset.filter(lambda x, y: tf.equal(y, 1))
        # 取第一条正样本（若存在）
        first_positive = positive_samples.take(1)
        # 取原始数据第一条（作为备选）
        first_any = window_dataset.take(1)
        # 优先返回正样本，否则返回原始第一条
        return first_positive.concatenate(first_any).take(1)

    # 应用分组逻辑（窗口大小设为足够大以包含同一用户的所有样本）
    return dataset.group_by_window(
        key_func=key_func,
        reduce_func=reduce_func,
        window_size=1000  # 根据实际场景调整
    )

def generator_dataset(data_path, batch_size, num_parallel_calls, is_test, use_group_by=False):
    """创建TFRecord数据集生成器"""
    # 获取文件列表（排除SUCCESS文件）
    files = glob.glob(data_path)
    files = list(filter(lambda x: x[-7:] != "SUCCESS", files))
    if not files:
        raise ValueError(f"未找到符合条件的文件: {data_path}")

    # 读取并解析TFRecord
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=num_parallel_calls)

    print("\n===== 处理前 =====")
    count_dataset(dataset)
    # 应用分组预处理（如需）
    if use_group_by:
        dataset = group_by_user_preprocessing(dataset)
    print("\n===== 处理后 =====")
    count_dataset(dataset)

    # 后续处理（测试集/训练集不同逻辑）
    if is_test:
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        dataset = dataset.shuffle(10000).repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset

def count_dataset(dataset, description="数据集"):
    """计算数据集的记录数"""
    count = 0
    for _ in dataset:
        count += 1
    print(f"{description}记录数: {count}")
    return count

def main():
    """主函数：生成数据、创建数据集并验证分组效果"""
    # 生成示例数据
    print("生成示例数据...")
    train_data = create_sample_data(5000)  # 生成5000条样本
    write_tfrecord(train_data, "train_group_data.tfrecord")

    # 创建带分组预处理的数据集
    train_dataset = generator_dataset(
        "train_group_data.tfrecord",
        batch_size=32,
        num_parallel_calls=tf.data.AUTOTUNE,
        is_test=False,
        use_group_by=True  # 启用分组预处理
    )

    # 验证分组结果
    print("\n验证分组预处理结果：")
    user_samples = {}  # 存储每个用户最终选中的样本label
    # 取部分样本查看（50个batch）
    for features, labels in train_dataset.take(50):
        user_ids = features["user_id"].numpy()
        for i in range(len(user_ids)):
            uid = user_ids[i]
            label = labels.numpy()[i]
            if uid not in user_samples:
                user_samples[uid] = label
                print(f"用户 {uid} 的选中样本label: {label}")

    # 统计结果
    total_users = len(user_samples)
    positive_count = sum(1 for label in user_samples.values() if label == 1)
    print(f"\n总用户数: {total_users}")
    print(f"选中正样本的用户数: {positive_count}")
    print(f"正样本比例: {positive_count / total_users:.2%}")

if __name__ == "__main__":
    main()