import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
import csv

# 创建一个包含缺失值的数据集
def load_csv(path):
    data_read = pd.read_csv(path, header=0)  # 加载数据，第一行作为列名
    data = data_read.values  # 获取数据
    var_names = data_read.columns.tolist()  # 获取列名
    print(data.shape)
    return data, var_names  # 返回数据和变量名


data1, var_names = load_csv("./SMC_change.csv")
data = pd.DataFrame(data1, columns =var_names)
print(data)

# 多重插补函数
def multiple_imputation(data, num_imputations=5):
    imputed_datasets = []

    for _ in range(num_imputations):
        # 使用迭代式插补方法
        imputer = IterativeImputer()
        imputed_data = imputer.fit_transform(data)
        imputed_data = pd.DataFrame(imputed_data, columns=data.columns)

        imputed_datasets.append(imputed_data)

    return imputed_datasets


# 单独分析参数的函数
def analyze_parameters(imputed_datasets):
    analyzed_parameters = []

    for dataset in imputed_datasets:
        # 使用线性回归模型进行拟合
        model = LinearRegression()
        observed_values = dataset.dropna(subset=['X1'], axis=0)['X1']
        features = dataset.drop(['X1'], axis=1)
        model.fit(features, observed_values)

        # 获取模型参数
        params = np.append(model.intercept_, model.coef_)
        analyzed_parameters.append(params)

    return analyzed_parameters


# 汇总结果估计的函数
def calculate_final_imputation(imputed_datasets):
    analyzed_parameters = analyze_parameters(imputed_datasets)
    final_imputation = pd.DataFrame(imputed_datasets[0].copy())

    for i, param in enumerate(np.mean(analyzed_parameters, axis=0)):
        final_imputation.iloc[:, i] = final_imputation.iloc[:, i].fillna(param)

    return final_imputation


# 进行多重插补
imputed_datasets = multiple_imputation(data)

# 计算汇总结果估计
final_imputation = calculate_final_imputation(imputed_datasets)
print("Final imputation:")
print(final_imputation)