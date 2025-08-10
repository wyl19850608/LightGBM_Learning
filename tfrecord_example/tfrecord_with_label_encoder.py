import tensorflow as tf
import glob
import numpy as np
from collections import defaultdict

# 全局变量定义
all_features = ["user_id", "age", "gender", "click_count", "purchase_count", "label", "history_items", "city"]
all_feature_types = ["int", "int", "string", "int", "int", "int", "ARRAY<int>", "string"]
model_features = ["user_id", "age", "gender", "click_count", "purchase_count", "history_items", "city"]

def create_sample_data(num_samples=1000):
    """生成示例数据，包含字符串特征用于演示LabelEncoder"""
    cities = ["北京", "上海", "广州", "深圳", "杭州", "成都"]
    genders = ["男", "女", "未知"]

    data = []
    for i in range(num_samples):
        has_positive = (i % 5 != 0)
        label = np.random.randint(0, 2) if has_positive else 0

        sample = {
            "user_id": i % 100,
            "age": np.random.randint(18, 70),
            "gender": np.random.choice(genders),
            "click_count": np.random.randint(0, 100),
            "purchase_count": np.random.randint(0, 20),
            "label": label,
            "history_items": np.random.randint(1000, 10000, size=10),
            "city": np.random.choice(cities)
        }
        data.append(sample)
    return data

def write_tfrecord(data, output_path):
    """将数据写入TFRecord文件，支持字符串特征"""
    writer = tf.io.TFRecordWriter(output_path)
    for sample in data:
        feature = {}
        # 处理整数特征
        for feature_name in ["user_id", "age", "click_count", "purchase_count", "label"]:
            feature[feature_name] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[sample[feature_name]])
            )
        # 处理字符串特征
        for feature_name in ["gender", "city"]:
            feature[feature_name] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[sample[feature_name].encode('utf-8')])
            )
        # 处理数组特征（历史物品）
        feature["history_items"] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=sample["history_items"])
        )
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()
    print(f"TFRecord文件已写入: {output_path}")

def read_tfrecord(example):
    """解析TFRecord文件中的样本，包括字符串特征"""
    format_builder = {}
    filter_feature_names = []

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

def get_categorical_mappings(dataset, cat_features):
    """从数据集中获取分类特征的映射关系（字符串到整数）"""
    mappings = {feat: defaultdict(int) for feat in cat_features}
    unique_values = {feat: set() for feat in cat_features}

    # 遍历数据集收集唯一值（使用Eager模式）
    for features, _ in dataset:
        for feat in cat_features:
            if feat in features:
                # 字符串需要解码
                if features[feat].dtype == tf.string:
                    # 这里可以用numpy()因为是在获取映射阶段，非图模式
                    value = features[feat].numpy().decode('utf-8')
                else:
                    value = features[feat].numpy()
                unique_values[feat].add(value)

    # 创建映射（从值到整数编码）
    for feat in cat_features:
        values = sorted(unique_values[feat])
        for idx, value in enumerate(values):
            mappings[feat][value] = idx
        print(f"特征 {feat} 的编码映射: {dict(mappings[feat])}")

    return mappings

def apply_label_encoding(features, label, mappings):
    """
    应用LabelEncoder转换（修复版）
    使用纯TensorFlow操作，避免使用.numpy()
    """
    encoded_features = features.copy()

    for feat, mapping in mappings.items():
        if feat in encoded_features:
            # 创建映射表的Tensor版本
            mapping_keys = tf.constant(list(mapping.keys()), dtype=tf.string)
            mapping_values = tf.constant(list(mapping.values()), dtype=tf.int32)
            num_classes = tf.shape(mapping_keys)[0]

            # 处理字符串特征
            if encoded_features[feat].dtype == tf.string:
                # 查找当前值在映射表中的索引
                # 使用tf.where找到匹配的位置
                matches = tf.where(tf.equal(mapping_keys, encoded_features[feat]))

                # 如果找到匹配项，使用对应的索引；否则使用num_classes（未知值）
                encoded_value = tf.cond(
                    tf.greater(tf.shape(matches)[0], 0),
                    lambda: tf.gather(mapping_values, matches[0, 0]),
                    lambda: num_classes
                )
            else:
                # 处理整数类型的分类特征
                # 转换为字符串进行比较（因为映射表的键是字符串）
                feat_str = tf.strings.as_string(encoded_features[feat])
                matches = tf.where(tf.equal(mapping_keys, feat_str))

                encoded_value = tf.cond(
                    tf.greater(tf.shape(matches)[0], 0),
                    lambda: tf.gather(mapping_values, matches[0, 0]),
                    lambda: num_classes
                )

            # 转换为int32并替换原特征
            encoded_features[feat] = tf.cast(encoded_value, tf.int32)
            # 添加未知值标识特征
            encoded_features[f"{feat}_is_unknown"] = tf.cast(
                tf.equal(encoded_value, num_classes),
                tf.int32
            )

    return encoded_features, label

def count_dataset(dataset, description="数据集"):
    """计算数据集的记录数"""
    count = 0
    for _ in dataset:
        count += 1
    print(f"{description}记录数: {count}")
    return count

def group_by_user_preprocessing(dataset):
    """按user_id分组，每组内优先选择第一条label=1的样本，若无则选择第一条样本"""
    print("\n===== 分组处理前 =====")
    original_count = count_dataset(dataset, "处理前")

    # 重新创建数据集（因为计数会消耗原始数据集）
    dataset_list = list(dataset.as_numpy_iterator())
    recreated_dataset = tf.data.Dataset.from_generator(
        lambda: iter(dataset_list),
        output_signature=dataset.element_spec
    )

    def key_func(x, y):
        return tf.cast(x["user_id"], tf.string)

    def reduce_func(key, window_dataset):
        """处理每个用户的样本窗口"""
        positive_samples = window_dataset.filter(lambda x, y: tf.equal(y, 1))
        first_positive = positive_samples.take(1)
        first_any = window_dataset.take(1)
        return first_positive.concatenate(first_any).take(1)

    grouped_dataset = recreated_dataset.group_by_window(
        key_func=key_func,
        reduce_func=reduce_func,
        window_size=1000
    )

    print("\n===== 分组处理后 =====")
    grouped_count = count_dataset(grouped_dataset, "处理后")
    print(f"估计用户数量: {grouped_count}")
    print(f"数据压缩比例: {grouped_count / original_count:.2%}")

    return grouped_dataset

def generator_dataset(data_path, batch_size, num_parallel_calls, is_test,
                      use_group_by=False, cat_features=None):
    """创建TFRecord数据集生成器，支持LabelEncoder"""
    files = glob.glob(data_path)
    files = list(filter(lambda x: x[-7:] != "SUCCESS", files))
    if not files:
        raise ValueError(f"未找到符合条件的文件: {data_path}")

    # 读取并解析TFRecord
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=num_parallel_calls)

    # 获取分类特征映射（仅在训练集且有分类特征时）
    mappings = None
    if cat_features and not is_test:
        print("\n===== 计算分类特征映射 =====")
        # 创建临时数据集用于计算映射
        temp_dataset = tf.data.TFRecordDataset(files)
        temp_dataset = temp_dataset.map(read_tfrecord, num_parallel_calls=num_parallel_calls)
        mappings = get_categorical_mappings(temp_dataset, cat_features)

        # 应用LabelEncoder
        from functools import partial
        encode_fn = partial(apply_label_encoding, mappings=mappings)
        dataset = dataset.map(encode_fn, num_parallel_calls=num_parallel_calls)
    elif cat_features and is_test and mappings is None:
        raise ValueError("测试集需要传入训练好的mappings")

    # 应用分组预处理（如需）
    if use_group_by:
        dataset = group_by_user_preprocessing(dataset)

    # 后续处理
    if is_test:
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        dataset = dataset.shuffle(10000).repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset, mappings

def main():
    """主函数：生成数据、创建数据集并验证功能"""
    print("生成示例数据...")
    train_data = create_sample_data(500)
    write_tfrecord(train_data, "train_group_data.tfrecord")

    # 定义需要编码的分类特征
    categorical_features = ["gender", "city"]

    # 创建带LabelEncoder和分组预处理的数据集
    train_dataset, mappings = generator_dataset(
        "train_group_data.tfrecord",
        batch_size=32,
        num_parallel_calls=tf.data.AUTOTUNE,
        is_test=False,
        use_group_by=True,
        cat_features=categorical_features
    )

    # 验证编码结果
    print("\n===== 验证编码结果 =====")
    for features, labels in train_dataset.take(1):
        print("编码后的gender特征:", features["gender"].numpy()[:50])
        print("gender是否为未知值:", features["gender_is_unknown"].numpy()[:50])
        print("编码后的city特征:", features["city"].numpy()[:50])
        print("city是否为未知值:", features["city_is_unknown"].numpy()[:50])

if __name__ == "__main__":
    main()
