import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Concatenate, Dropout, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

# ================ 1. 数据处理工具类（增加NaN处理） ================
class DataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scalers = {}
        self.string_categories = {}
        self.field_dims = []  # 只记录需要嵌入的特征维度
        self.numeric_indices = []  # 记录连续特征的索引位置
        self.fill_values = {}  # 存储各特征的缺失值填充值

    def fit(self, df, numeric_cols, categorical_cols, bool_cols, string_cols, label_col):
        """拟合数据处理器，学习数据的转换规则，包括缺失值处理"""
        all_features = []
        self.numeric_indices = list(range(len(numeric_cols)))

        # 处理连续型数字特征 - 用均值填充NaN
        for col in numeric_cols:
            # 计算填充值并存储
            fill_val = df[col].mean()
            self.fill_values[col] = fill_val
            # 填充缺失值
            df[col] = df[col].fillna(fill_val)

            scaler = StandardScaler()
            df[col] = scaler.fit_transform(df[[col]])
            self.scalers[col] = scaler
            all_features.append(col)

        # 处理非连续型数字特征 - 用最频繁值填充NaN
        for col in categorical_cols:
            df[col] = df[col].astype(str)
            # 计算填充值并存储
            fill_val = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
            self.fill_values[col] = fill_val
            # 填充缺失值
            df[col] = df[col].fillna(fill_val)

            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
            self.field_dims.append(len(le.classes_))
            all_features.append(col)

        # 处理布尔型特征 - 用最频繁值填充NaN
        for col in bool_cols:
            # 计算填充值并存储
            fill_val = df[col].mode()[0] if not df[col].mode().empty else 0
            self.fill_values[col] = fill_val
            # 填充缺失值
            df[col] = df[col].fillna(fill_val).astype(int)

            self.field_dims.append(2)  # 布尔特征有2个可能值
            all_features.append(col)

        # 处理字符串特征 - 用最频繁值或'unknown'填充NaN
        for col in string_cols:
            # 计算填充值并存储
            fill_val = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
            self.fill_values[col] = fill_val
            # 填充缺失值
            df[col] = df[col].fillna(fill_val)

            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.string_categories[col] = le.classes_
            self.field_dims.append(len(le.classes_))
            all_features.append(col)

        # 处理标签列
        if label_col:
            # 处理标签列的缺失值
            if df[label_col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[label_col]):
                    fill_val = df[label_col].mean()
                else:
                    fill_val = df[label_col].mode()[0] if not df[label_col].mode().empty else 'unknown'
                self.fill_values[label_col] = fill_val
                df[label_col] = df[label_col].fillna(fill_val)

            if df[label_col].dtype == 'object':
                le = LabelEncoder()
                df[label_col] = le.fit_transform(df[label_col])
                self.label_encoders[label_col] = le

        return df, all_features

    def transform(self, df, numeric_cols, categorical_cols, bool_cols, string_cols, label_col=None):
        """应用学习到的转换规则处理数据，包括缺失值填充"""
        # 处理连续型数字特征
        for col in numeric_cols:
            if col in self.scalers and col in self.fill_values:
                # 填充缺失值
                df[col] = df[col].fillna(self.fill_values[col])
                # 标准化
                df[col] = self.scalers[col].transform(df[[col]])

        # 处理非连续型数字特征
        for col in categorical_cols:
            if col in self.label_encoders and col in self.fill_values:
                df[col] = df[col].astype(str)
                # 填充缺失值
                df[col] = df[col].fillna(self.fill_values[col])
                # 处理未知类别
                known_classes = set(self.label_encoders[col].classes_)
                df[col] = df[col].apply(lambda x: x if x in known_classes else 'unknown')
                # 编码
                df[col] = self.label_encoders[col].transform(df[col])

        # 处理布尔型特征
        for col in bool_cols:
            if col in self.fill_values:
                # 填充缺失值
                df[col] = df[col].fillna(self.fill_values[col]).astype(int)

        # 处理字符串特征
        for col in string_cols:
            if col in self.string_categories and col in self.fill_values:
                # 填充缺失值
                df[col] = df[col].fillna(self.fill_values[col])
                df[col] = df[col].astype(str)
                # 处理未知类别
                known_classes = set(self.string_categories[col])
                df[col] = df[col].apply(lambda x: x if x in known_classes else 'unknown')
                le = LabelEncoder()
                le.classes_ = np.append(self.string_categories[col], 'unknown')
                df[col] = le.transform(df[col])

        # 处理标签列
        if label_col and label_col in self.fill_values:
            # 填充缺失值
            df[label_col] = df[label_col].fillna(self.fill_values[label_col])
            if label_col in self.label_encoders:
                df[label_col] = self.label_encoders[label_col].transform(df[label_col])

        return df

# ================ 2. DeepFM模型组件 ================
class FM(Layer):
    """Factorization Machine层，计算二阶交互特征"""
    def __init__(self, **kwargs):
        super(FM, self).__init__(** kwargs)

    def call(self, inputs):
        # 输入格式: [batch_size, field_num, embedding_dim]
        square_of_sum = tf.square(tf.reduce_sum(inputs, axis=1, keepdims=True))
        sum_of_square = tf.reduce_sum(tf.square(inputs), axis=1, keepdims=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2, keepdims=False)
        return cross_term

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

class DeepFM:
    def __init__(self, num_numeric, field_dims, embedding_dim=8, hidden_units=[128, 64],
                 dropout_rate=0.5, l2_reg=0.0):
        """初始化DeepFM模型"""
        self.num_numeric = num_numeric
        self.field_dims = field_dims
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg

    def build_model(self):
        """构建DeepFM模型"""
        # 输入层
        inputs = []

        # 连续特征输入（不使用嵌入层）
        numeric_inputs = []
        for i in range(self.num_numeric):
            num_input = Input(shape=(1,), name=f'numeric_input_{i}')
            numeric_inputs.append(num_input)
            inputs.append(num_input)

        # 离散特征输入（使用嵌入层）
        embedding_inputs = []
        linear_embeddings = []  # 一阶特征
        feature_embeddings = []  # 二阶嵌入特征

        for i, dim in enumerate(self.field_dims):
            input_layer = Input(shape=(1,), name=f'embedding_input_{i}')
            embedding_inputs.append(input_layer)
            inputs.append(input_layer)

            # 一阶权重
            linear_embedding = Embedding(dim, 1, embeddings_regularizer=l2(self.l2_reg))(input_layer)
            linear_embeddings.append(Flatten()(linear_embedding))

            # 二阶嵌入
            feature_embedding = Embedding(dim, self.embedding_dim, embeddings_regularizer=l2(self.l2_reg))(input_layer)
            feature_embeddings.append(feature_embedding)

        # FM部分 - 一阶项
        if linear_embeddings:
            discrete_linear = Concatenate(axis=1)(linear_embeddings)
            linear_part = Concatenate(axis=1)(numeric_inputs + [discrete_linear])
        else:
            linear_part = Concatenate(axis=1)(numeric_inputs)
        linear_output = Dense(1)(linear_part)

        # FM部分 - 二阶项（只针对离散特征）
        fm_output = tf.zeros_like(linear_output)  # 默认值
        if feature_embeddings:
            fm_input = Concatenate(axis=1)(feature_embeddings)
            fm_output = FM()(fm_input)

        # Deep部分
        deep_inputs = numeric_inputs.copy()
        if feature_embeddings:
            deep_inputs.append(Flatten()(fm_input))

        deep_input = Concatenate()(deep_inputs)
        for units in self.hidden_units:
            deep_input = Dense(units, activation='relu', kernel_regularizer=l2(self.l2_reg))(deep_input)
            deep_input = Dropout(self.dropout_rate)(deep_input)
        deep_output = Dense(1)(deep_input)

        # 组合输出
        concat_output = Concatenate()([linear_output, fm_output, deep_output])
        output = Dense(1, activation='sigmoid')(concat_output)

        # 创建模型
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

# ================ 3. 主函数示例（包含NaN值） ================
def main():
    # 示例：创建包含NaN值的模拟数据
    np.random.seed(42)
    n_samples = 10000

    # 创建示例数据框，添加5%左右的NaN值
    data = {
        # 连续型特征 - 包含NaN值
        'num_feature1': np.where(np.random.random(n_samples) < 0.05, np.nan, np.random.normal(0, 1, n_samples)),
        'num_feature2': np.where(np.random.random(n_samples) < 0.03, np.nan, np.random.normal(5, 2, n_samples)),

        # 非连续型数字特征 - 包含NaN值
        'cat_feature1': np.where(np.random.random(n_samples) < 0.04, np.nan, np.random.randint(0, 5, n_samples)),
        'cat_feature2': np.where(np.random.random(n_samples) < 0.06, np.nan, np.random.randint(0, 3, n_samples)),

        # 布尔型特征 - 包含NaN值
        'bool_feature1': np.where(np.random.random(n_samples) < 0.02, np.nan, np.random.randint(0, 2, n_samples).astype(bool)),
        'bool_feature2': np.where(np.random.random(n_samples) < 0.07, np.nan, np.random.randint(0, 2, n_samples).astype(bool)),

        # 字符串特征 - 包含NaN值
        'str_feature1': np.where(
            np.random.random(n_samples) < 0.05,
            np.nan,
            np.random.choice(['apple', 'banana', 'cherry', 'date'], n_samples)
        ),
        'str_feature2': np.where(
            np.random.random(n_samples) < 0.04,
            np.nan,
            np.random.choice(['red', 'green', 'blue', 'yellow', 'black'], n_samples)
        ),

        # 标签 - 包含NaN值
        'label': np.where(np.random.random(n_samples) < 0.03, np.nan, np.random.randint(0, 2, n_samples))
    }

    df = pd.DataFrame(data)

    # 打印原始数据中的NaN值统计
    print("原始数据中的NaN值统计:")
    print(df.isnull().sum())

    # 定义字段类型
    numeric_cols = ['num_feature1', 'num_feature2']
    categorical_cols = ['cat_feature1', 'cat_feature2']
    bool_cols = ['bool_feature1', 'bool_feature2']
    string_cols = ['str_feature1', 'str_feature2']
    label_col = 'label'

    # 数据预处理
    processor = DataProcessor()
    df_processed, all_features = processor.fit(df, numeric_cols, categorical_cols, bool_cols, string_cols, label_col)

    # 确认处理后的数据中没有NaN值
    print("\n处理后的NaN值统计:")
    print(df_processed.isnull().sum())

    print("df.row ", df.count())

    # 准备训练数据
    X = np.column_stack([df_processed[col].values for col in all_features])
    y = df_processed[label_col].values

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 将训练集和测试集重新拆分为特征列表
    X_train = [X_train[:, i] for i in range(X_train.shape[1])]
    X_test = [X_test[:, i] for i in range(X_test.shape[1])]

    # 构建并训练模型
    num_numeric = len(numeric_cols)
    model_builder = DeepFM(
        num_numeric=num_numeric,
        field_dims=processor.field_dims,
        embedding_dim=8,
        hidden_units=[128, 64],
        dropout_rate=0.3,
        l2_reg=0.001
    )
    model = model_builder.build_model()

    # 打印模型结构
    model.summary()

    # 训练模型
    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=256,
        validation_split=0.1,
        verbose=1
    )

    # 评估模型
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    return model, processor

if __name__ == "__main__":
    model, processor = main()