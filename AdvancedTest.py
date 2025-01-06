import numpy as np  # 导入NumPy库，用于数值计算
import pandas as pd  # 导入Pandas库，用于数据处理
from sklearn.model_selection import train_test_split  # 导入train_test_split函数，用于拆分数据集
from sklearn.linear_model import LinearRegression  # 导入LinearRegression类，用于线性回归模型
from sklearn.metrics import mean_squared_error, r2_score  # 导入评估指标

import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图

# 生成一些合成数据
np.random.seed(42)  # 设置随机种子以确保结果可重复
X = 2 * np.random.rand(100, 1)  # 生成100个随机数作为特征
y = 4 + 3 * X + np.random.randn(100, 1)  # 生成目标值，带有一些噪声

# 将数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80%训练，20%测试

# 创建线性回归模型
model = LinearRegression()  # 实例化线性回归模型
model.fit(X_train, y_train)  # 在训练集上训练模型

# 进行预测
y_pred = model.predict(X_test)  # 在测试集上进行预测

# 评估模型
mse = mean_squared_error(y_test, y_pred)  # 计算均方误差
r2 = r2_score(y_test, y_pred)  # 计算R^2得分

print(f"Mean Squared Error: {mse}")  # 输出均方误差
print(f"R^2 Score: {r2}")  # 输出R^2得分

# 绘制结果
plt.scatter(X_test, y_test, color='black', label='Actual data')  # 绘制实际数据点
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Predicted line')  # 绘制预测线
plt.xlabel('X')  # 设置X轴标签
plt.ylabel('y')  # 设置Y轴标签
plt.title('Linear Regression Example')  # 设置图表标题
plt.legend()  # 显示图例
plt.show()  # 显示图表