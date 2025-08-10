import os

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据路径配置
DATA_CONFIG = {
    "raw_train_data": os.path.join(ROOT_DIR, "data", "raw", "raw_train_data.csv"),
    "processed_train_data": os.path.join(ROOT_DIR, "data", "processed", "train"),
    "pred_input_data": os.path.join(ROOT_DIR, "data", "input", "pred_data.csv"),
    "pred_output_data": os.path.join(ROOT_DIR, "data", "output", "pred_results"),
}

# 模型和元数据路径
MODEL_CONFIG = {
    "meta_data": os.path.join(ROOT_DIR, "models", "cat_meta_data.pkl"),
    "model_path": os.path.join(ROOT_DIR, "models", "lightgbm_model.txt"),
}

# Spark配置
SPARK_CONFIG = {
    "appName": "LightGBMWorkflow",
    "master": "local[*]",
    "driver_memory": "16g",
    "executor_memory": "16g",
    "arrow_enabled": "false"
}

# LightGBM训练参数
LIGHTGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "verbose": 1,
    "num_boost_round": 100,
}

# 数据生成参数
DATA_GEN_CONFIG = {
    "sample_size": 5,
    "generate_sample_size": 10000  # 生成的训练样本量
}