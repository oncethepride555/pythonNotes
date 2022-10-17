import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# g(z)
def sigmoid(z):  
    return 1 / (1 + np.exp(-z))

# 读入训练数据
def loadData(path):
    trainingData = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])

    # trainingData.head()
    # trainingData.describe()

    positive = trainingData[trainingData['Admitted'].isin([1])] #把Admitted列中值为1的所有行选出来
    negative = trainingData[trainingData['Admitted'].isin([0])]

    fig, ax = plt.subplots(figsize=(8,5)) #创建一个8*5的画布
    ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
    ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
    ax.legend() #图例
    ax.set_xlabel('Exam 1 Score') # x轴名称
    ax.set_ylabel('Exam 2 Score')
    plt.show()
    return trainingData

# J(θ)
# 计算损失，用了矢量化编程而不是for循环
def computeLoss(X, y, theta):  
    theta = np.copy(theta)
    X = np.copy(X)
    y = np.copy(y)
    m = X.shape[0] # X的行数，即有多少组数据
    h = sigmoid(np.matmul(X, theta.T))
    first = np.matmul(-(y.T), np.log(h))
    second = np.matmul((1 - y).T, np.log(1 - h))
    return np.sum(first - second) / m

# 梯度下降部分
def gradientDescent(X, y, theta, alpha, iters): # iters迭代次数
    m = X.shape[0] # 数据项数m，有m条数据
    temp = np.matrix(np.zeros(theta.shape)) # [[0. 0. 0.]]
    # parameters = 1
    cost = np.zeros(iters) # [0. 0. ... 0. 0.] iters个0的数组

    for i in range(iters):
        error = sigmoid(np.matmul(X, theta.T)) - y
        theta = theta - ((alpha/m) * np.matmul(X.T, error)).T
        cost[i] = computeLoss(X, y, theta)

    return theta, cost

def predict(theta, X):  
    probability = sigmoid(np.matmul(X, theta.T))
    return [1 if x >= 0.5 else 0 for x in probability] # g(x)>=0.5,y=1

trainingData = loadData(os.getcwd() + '/机器学习/data/ex2data1.txt')

# 插入常数项
trainingData.insert(0, 'Ones', 1)

cols = trainingData.shape[1]  # 数据集的列数，包含了插入的常数项的那一列，一共4列
X = trainingData.iloc[:,0:cols-1] # 第1，2，3列所有行
y = trainingData.iloc[:,cols-1:cols] # 第4列所有行

# 初始化X、Y以及theta矩阵
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.zeros(3)) # [[0. 0. 0.]]  【θ0x0 + θ1x1 + θ2x2】其中x0=1

# 计算训练前的损失值
computeLoss(X, y, theta)

# 使用梯度下降得到模型参数
alpha = 0.001
iters = 2000000
theta_fin, loss = gradientDescent(X, y, theta, alpha, iters)
# print(theta_fin) # [[-0.13839333  0.01138625  0.00152741]]


# 计算训练后的参数的损失值 (不优化)
computeLoss(X, y, theta_fin)
print(theta_fin.T[0])

# 为了画线用的，画出训练好后的直线
# print(X[:,1])
x1 = np.linspace(X[:,1].min(),X[:,1].max(),100) #第三个参数是生成的样本的数量
# print(x1)
x2 = (-theta_fin[0,0]-theta_fin[0,1]*x1)/theta_fin[0,2]
fig, ax = plt.subplots(figsize=(8,5))  # 创建一个8*5的画布
ax.plot(x1, x2, 'r', label='Prediction')
positive = trainingData[trainingData['Admitted'].isin([1])] #把Admitted列中值为1的所有行选出来
negative = trainingData[trainingData['Admitted'].isin([0])]
# print(positive)
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend() #图例
ax.set_xlabel('x1') # x轴名称
ax.set_ylabel('x2')
plt.show()


# 损失随着迭代次数的变化 (不优化)
fig, ax = plt.subplots(figsize=(8,5))  
ax.plot(np.arange(iters), loss, 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Error vs. Training Epoch') 
plt.show()

# 不理解为什么不优化的会这么低，是学习速率没动态变化么？
predictions = predict(theta_fin, X)  
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]  
accuracy = (sum(map(int, correct)) % len(correct))  
print('accuracy 1 = {0}%'.format(accuracy)) # 65%


# # 使用scipy的optimize来做优化
# import scipy.optimize as opt
# # 换了下参数位置让其符合fmin_tnc
# def gradient(theta, X, y):
#     theta = np.matrix(theta)
#     X = np.matrix(X)
#     y = np.matrix(y)
    
#     parameters = int(theta.ravel().shape[1])
#     grad = np.zeros(parameters)
    
#     error = sigmoid(X * theta.T) - y
    
#     for i in range(parameters):
#         term = np.multiply(error, X[:,i])
#         grad[i] = np.sum(term) / len(X)
    
#     return grad
# # 换了下参数位置让其符合fmin_tnc
# def computeLoss2(theta, X, y):  
#     theta = np.copy(theta)
#     X = np.copy(X)
#     y = np.copy(y)
#     m = X.shape[0]
#     h = sigmoid(np.matmul(X, theta.T))
#     first = np.matmul(-(y.T), np.log(h))
#     second = np.matmul((1 - y).T, np.log(1 - h))
#     return np.sum(first - second) / m
# result = opt.fmin_tnc(func=computeLoss2, x0=theta, fprime=gradient, args=(X, y))

# predictions = predict(np.matrix(result[0]), X)  
# correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]  
# accuracy = (sum(map(int, correct)) % len(correct))  
# print('accuracy 2 = {0}%'.format(accuracy)) # 89%