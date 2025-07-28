# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 指定后端，避免PyCharm兼容性问题
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns

# 设置中文显示
plt.rcParams["font.family"] = ["Heiti TC", "STHeiti", "Arial Unicode MS"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 生成模拟数据
def generate_data(n_samples=1000, seed=42):
    np.random.seed(seed)

    # 生成特征
    X = np.random.normal(0, 1, size=(n_samples, 5))  # 5个特征

    # 生成干预变量（二分类）
    # 高价值用户(x1>0)更可能被干预
    propensity = 1 / (1 + np.exp(-(X[:, 0])))
    T = np.random.binomial(1, propensity)

    # 生成潜在结果（因果效应与x1和x2相关）
    y0 = 2 * X[:, 0] + 3 * X[:, 1] + np.random.normal(0, 1, n_samples)  # 未干预结果
    y1 = y0 + 1 + 0.5 * X[:, 1]  # 干预结果（因果效应：1 + 0.5*x2）

    # 观测结果
    Y = T * y1 + (1 - T) * y0

    # 真实个体因果效应(ITE)
    true_ite = y1 - y0

    # 转换为DataFrame
    df = pd.DataFrame(X, columns=['x{}'.format(i) for i in range(1, 6)])
    df['T'] = T
    df['Y'] = Y
    df['true_ite'] = true_ite

    return df

# 2. 实现S-learner
def s_learner(X, T, Y):
    """
    S-learner实现：将干预变量作为特征加入模型

    参数:
    X: 特征矩阵
    T: 干预变量
    Y: 观测结果

    返回:
    model: 训练好的模型
    """
    # 合并特征和干预变量
    X_with_T = np.hstack([X, T.reshape(-1, 1)])

    # 使用随机森林作为基础模型
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_with_T, Y)

    return model

# 3. 预测个体因果效应(ITE)
def predict_ite(model, X):
    """
    预测个体因果效应

    参数:
    model: 训练好的S-learner模型
    X: 特征矩阵

    返回:
    predicted_ite: 预测的个体因果效应
    """
    # 预测干预组结果
    X_T1 = np.hstack([X, np.ones((X.shape[0], 1))])
    y_pred_1 = model.predict(X_T1)

    # 预测对照组结果
    X_T0 = np.hstack([X, np.zeros((X.shape[0], 1))])
    y_pred_0 = model.predict(X_T0)

    # 计算ITE
    predicted_ite = y_pred_1 - y_pred_0

    return predicted_ite

# 4. 评估模型性能
def evaluate_model(true_ite, predicted_ite):
    """
    评估模型预测的ITE与真实ITE的差异

    参数:
    true_ite: 真实个体因果效应
    predicted_ite: 预测的个体因果效应

    返回:
    rmse: 均方根误差
    """
    rmse = np.sqrt(mean_squared_error(true_ite, predicted_ite))
    return rmse

# 5. 主函数：运行S-learner并可视化结果
def main():
    # 生成数据
    df = generate_data(n_samples=1000)

    # 划分特征、干预和结果
    X = df[['x{}'.format(i) for i in range(1, 6)]].values
    T = df['T'].values
    Y = df['Y'].values
    true_ite = df['true_ite'].values

    # 划分训练集和测试集
    X_train, X_test, T_train, T_test, Y_train, Y_test, true_ite_train, true_ite_test = train_test_split(
        X, T, Y, true_ite, test_size=0.3, random_state=42
    )

    # 训练S-learner
    model = s_learner(X_train, T_train, Y_train)

    # 预测ITE
    predicted_ite = predict_ite(model, X_test)

    # 评估模型
    rmse = evaluate_model(true_ite_test, predicted_ite)
    print('S-learner RMSE: {:.4f}'.format(rmse))

    # 可视化预测结果与真实结果的关系
    plt.figure(figsize=(10, 6))
    plt.scatter(true_ite_test, predicted_ite, alpha=0.5)
    plt.plot([min(true_ite_test), max(true_ite_test)],
             [min(true_ite_test), max(true_ite_test)],
             'r--', label='理想情况')
    plt.xlabel('真实ITE')
    plt.ylabel('预测ITE')
    plt.title('S-learner预测的个体因果效应与真实值对比')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存图像而非显示（避免后端问题）
    plt.savefig('ite_comparison.png')
    plt.close()

    # 可视化特征重要性
    plt.figure(figsize=(10, 6))
    feature_importances = model.feature_importances_
    features = ['x{}'.format(i) for i in range(1, 6)] + ['T']
    plt.barh(features, feature_importances)
    plt.xlabel('特征重要性')
    plt.title('S-learner模型的特征重要性')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

    # 按分位数分析平均因果效应(ATE)
    df_test = pd.DataFrame(X_test, columns=['x{}'.format(i) for i in range(1, 6)])
    df_test['true_ite'] = true_ite_test
    df_test['predicted_ite'] = predicted_ite

    # 计算真实和预测的分位数因果效应
    df_test['quantile'] = pd.qcut(df_test['predicted_ite'], 5)
    ate_by_quantile = df_test.groupby('quantile').agg(
        true_ate=('true_ite', 'mean'),
        predicted_ate=('predicted_ite', 'mean')
    ).reset_index()

    # 可视化分位数因果效应
    plt.figure(figsize=(12, 6))
    x = range(len(ate_by_quantile))
    width = 0.35
    plt.bar(x, ate_by_quantile['true_ate'], width, label='真实ATE')
    plt.bar([i + width for i in x], ate_by_quantile['predicted_ate'], width, label='预测ATE')
    plt.xlabel('预测ITE分位数')
    plt.ylabel('平均因果效应(ATE)')
    plt.title('按预测ITE分位数分组的平均因果效应')
    plt.xticks([i + width/2 for i in x], ['Q{}'.format(i+1) for i in range(len(ate_by_quantile))])
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('ate_by_quantile.png')
    plt.close()

    print("图像已保存为：ite_comparison.png, feature_importance.png, ate_by_quantile.png")

if __name__ == "__main__":
    main()