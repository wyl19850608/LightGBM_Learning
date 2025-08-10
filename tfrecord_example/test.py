import tensorflow as tf
import numpy as np

# ------------------------------
# 1. 数据生成与TFRecord处理函数
# ------------------------------
def create_test_data(num_samples=5000):
    """生成带重复用户ID的测试数据（user_id为字符串类型）"""
    data = []
    for i in range(num_samples):
        # 每5个用户中有1个全是负样本
        has_positive = (i % 5 != 0)
        label = np.random.randint(0, 2) if has_positive else 0

        sample = {
            "user_id": f"user_{i % 100}",  # 字符串类型，共100个唯一用户
            "age": np.random.randint(18, 70),
            "gender": np.random.randint(0, 2),
            "label": label,
            "history_items": np.random.randint(1000, 10000, size=10)
        }
        data.append(sample)
    return data

def write_test_tfrecord(data, output_path):
    """将测试数据写入TFRecord"""
    with tf.io.TFRecordWriter(output_path) as writer:
        for sample in data:
            feature = {
                "user_id": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[sample["user_id"].encode('utf-8')])
                ),
                "age": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[sample["age"]])
                ),
                "gender": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[sample["gender"]])
                ),
                "label": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[sample["label"]])
                ),
                "history_items": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=sample["history_items"])
                )
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
    print(f"测试TFRecord已写入: {output_path}")

def parse_test_tfrecord(example_proto):
    """解析测试数据的TFRecord"""
    feature_desc = {
        "user_id": tf.io.FixedLenFeature([], tf.string),
        "age": tf.io.FixedLenFeature([], tf.int64),
        "gender": tf.io.FixedLenFeature([], tf.int64),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "history_items": tf.io.FixedLenFeature([10], tf.int64)
    }
    parsed = tf.io.parse_single_example(example_proto, feature_desc)

    # 转换类型并返回(特征, 标签)
    return {
        "user_id": parsed["user_id"],
        "age": tf.cast(parsed["age"], tf.int32),
        "gender": tf.cast(parsed["gender"], tf.int32),
        "history_items": tf.cast(parsed["history_items"], tf.int32)
    }, parsed["label"]

# ------------------------------
# 2. 分组处理函数（复用目标函数）
# ------------------------------
def group_by_user_preprocessing(dataset):
    """亿级用户分组核心逻辑"""
    def key_func(x, y):
        # 关键修改：用 tf.cond 替代 if 条件判断，适应图模式
        return tf.cond(
            tf.equal(y, -1),
            lambda: tf.constant(-1, dtype=tf.int64),
            lambda: tf.strings.to_hash_bucket_strong(
                x["user_id"],
                num_buckets=10**12,
                seed=[123456, 789012]
            )
        )

    def reduce_func(key, window_dataset):
        valid_samples = window_dataset.filter(lambda x, y: tf.not_equal(y, -1))

        # 按原始user_id去重
        def get_user_id(x, y):
            return x["user_id"]
        unique_samples = valid_samples.apply(
            tf.data.experimental.uniq_by(get_user_id)
        )

        # 优先选择正样本
        positive_samples = unique_samples.filter(lambda x, y: tf.equal(y, 1))
        first_positive = positive_samples.take(1)
        first_any = unique_samples.take(1)
        return first_positive.concatenate(first_any).take(1)

    return dataset.group_by_window(
        key_func=key_func,
        reduce_func=reduce_func,
        window_size=100000
    )

# ------------------------------
# 3. 测试主函数
# ------------------------------
def test_group_by_user():
    # 步骤1：生成测试数据并写入TFRecord
    test_data = create_test_data(num_samples=5000)
    tfrecord_path = "test_group_data.tfrecord"
    write_test_tfrecord(test_data, tfrecord_path)

    # 步骤2：读取并解析TFRecord为Dataset
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_test_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

    # 步骤3：应用分组预处理
    grouped_dataset = group_by_user_preprocessing(dataset)

    # 步骤4：验证分组结果
    print("\n===== 分组结果验证 =====")
    user_set = set()  # 存储已出现的用户ID
    positive_count = 0  # 统计选中正样本的用户数

    for features, label in grouped_dataset:
        user_id = features["user_id"].numpy().decode('utf-8')

        # 检查是否有重复用户（验证去重效果）
        if user_id in user_set:
            print(f"警告：用户 {user_id} 被重复分组！")
        user_set.add(user_id)

        # 记录正样本
        if label.numpy() == 1:
            positive_count += 1

    # 输出统计结果
    total_users = len(user_set)
    print(f"\n总唯一用户数: {total_users}")
    print(f"选中正样本的用户数: {positive_count}")
    print(f"正样本比例: {positive_count / total_users:.2%}")
    print(f"是否所有用户都被分组: {total_users == 100}")  # 预期100个唯一用户

if __name__ == "__main__":
    test_group_by_user()