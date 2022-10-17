# 参考http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-1/
# 用线性回归模型并训练出参数来预测我如果给另一个城市(城市人数)，那么卡车的利润是多少。
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 计算损失，用了矢量化编程而不是for循环
def computeLoss(X, y, theta):  
    inner = np.power(((X * theta.T) - y), 2) # np.power求幂
    return np.sum(inner) / (2 * len(X)) # len(X) = 97

# 梯度下降部分  alpha：学习速率   iters：迭代的次数
def gradientDescent(X, y, theta, alpha, iters):  
    temp = np.matrix(np.zeros(theta.shape)) # shape 行数和列数  得到的是与θ同行同列数的全0矩阵   # [[0. 0.]]
    parameters = int(theta.ravel().shape[1]) # 2
    cost = np.zeros(iters) # [0. 0. 0. ... 0. 0. 0.]

    for i in range(iters):
        
        error = (X * theta.T) - y
        for j in range(parameters):
            term = np.multiply(error, X[:,j]) # 矩阵对应元素位置相乘
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeLoss(X, y, theta)

    return theta, cost

# 读入训练数据
# windows用户路径可能需要修改下，后期有时间可能会做统一
def loadData(path):
    # 读取指定路径的文件，header=None 原始文件数据没有列索引，names 指定新的列索引
    trainingData = pd.read_csv(path, header=None, names=['Population', 'Profit'])

    trainingData.head() # 参数 n 默认为 5 ，得到的是前5行数据
#        Population   Profit
# 0      6.1101  17.5920
# 1      5.5277   9.1302
# 2      8.5186  13.6620
# 3      7.0032  11.8540
# 4      5.8598   6.8233

    trainingData.describe() # 对数据的一些分析
#            Population     Profit
# count   97.000000  97.000000
# mean     8.159800   5.839135
# std      3.869884   5.510262
# min      5.026900  -2.680700
# 25%      5.707700   1.986900
# 50%      6.589400   4.562300
# 75%      8.578100   7.046700
# max     22.203000  24.147000

    # figsize 图像窗口的大小
    trainingData.plot(kind='scatter', x='Population', y='Profit', figsize=(8,5))
    # bar或barh 条形
    # hist 直方图 是一种可以对值频率离散化显示的柱状图
    # scatter 散点图
    # line 折线图
    # kde 密度图 与直方图相关的一种类型图，是通过计算“可能会产生观测数据的连续概率分布的估计”而产生的
    plt.show()
    return trainingData
# os.getcwd() 方法用于返回当前工作目录
# print(os.getcwd())   E:\python_workspace\机器学习
trainingData = loadData(os.getcwd()+'\\机器学习\\data\\ex1data1.txt')

# 在数据集前插入一列Ones作为常数系数，也就是y=k*x+b*1这种形式
trainingData.insert(0, 'Ones', 1)
#    Ones  Population    Profit
# 0      1      6.1101  17.59200
# 1      1      5.5277   9.13020
# 2      1      8.5186  13.66200
# 3      1      7.0032  11.85400
# 4      1      5.8598   6.82330
# ..   ...         ...       ...
# 92     1      5.8707   7.20290
# 93     1      5.3054   1.98690
# 94     1      8.2934   0.14454
# 95     1     13.3940   9.05510
# 96     1      5.4369   0.61705

# 将输入X以及输出y从数据集中分割
cols = trainingData.shape[1]  # 3列 
X = trainingData.iloc[:,0:cols-1]  # 所有行，第0到2列，不包含第二列
y = trainingData.iloc[:,cols-1:cols]  # 所有行，第2到3列，不包含第三列

# 把pandas的DataFrames转换成numpy的矩阵
X = np.matrix(X.values)  
# [[ 1.      6.1101]
#  [ 1.      5.5277]
#  [ 1.      8.5186]
#  [ 1.      7.0032]
#  ...              ]
y = np.matrix(y.values)  
# 初始化参数为全0的，当然也可以初始化成其他的
theta = np.matrix(np.array([0,0]))  # [[0 0]]

# 各向量的维度
X.shape, theta.shape, y.shape  
# X.shape (97, 2)  theta.shape (1,2)  y.shape(97,1)
# 初始损失函数值
computeLoss(X, y, theta)   # 32.07，后面可以看看训练完后的损失函数值

# 设置学习速率以及迭代次数
alpha = 0.01  
iters = 2000

# 使用梯度下降得到模型参数
theta_fin, loss = gradientDescent(X, y, theta, alpha, iters)  
theta_fin

# 计算训练后的参数的损失值
computeLoss(X, y, theta_fin)  # 4.47

# 为了画线用的，画出训练好后的直线
x = np.linspace(trainingData.Population.min(), trainingData.Population.max(), 100)  
f = theta_fin[0, 0] + (theta_fin[0, 1] * x)

fig, ax = plt.subplots(figsize=(8,5))  # 创建一个8*5的画布
ax.plot(x, f, 'r', label='Prediction')  # r red简写；label='Prediction'显示在图例上
ax.scatter(trainingData.Population, trainingData.Profit, label='Traning Data')  
ax.legend(loc=2)  # .legend 给图像添加图例
ax.set_xlabel('Population')  # x轴标注
ax.set_ylabel('Profit')  # y轴标注
ax.set_title('Predicted Profit vs. Population Size')  # 标题
plt.show()

# 损失随着迭代次数的变化
fig, ax = plt.subplots(figsize=(8,5))  
ax.plot(np.arange(iters), loss, 'r')  # loss是不同迭代次数时 损失函数的值
ax.set_xlabel('Iterations')  
ax.set_ylabel('Loss')  
ax.set_title('Error vs. Training Epoch')  
plt.show()