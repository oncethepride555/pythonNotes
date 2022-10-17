import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

z = np.arange(-10,10,0.1)
p = sigmoid(z)
plt.plot(z,p) # plt.plot()实际上会通过plt.gca()获得当前的Axes对象ax，然后再调用ax.plot()方法实现真正的绘图
plt.axvline(color='k') # 垂直的线，默认x=0
plt.axhspan(0.0,1.0,facecolor='0.7',alpha=0.4) # 暗框
plt.axhline(y=0.5,color='k',ls='dotted') # 水平线，dotted 虚线
plt.axhline(y=0.0,color='0.4',ls='dotted')
plt.axhline(y=1.0,color='0.4',ls='dotted')
plt.ylim(-0.1,1.1) # y轴整个的范围
plt.yticks([0.0,0.5,1.0]) # y轴显示的刻度
plt.ylabel('g(z)') # y轴标题
plt.xlabel('z') # x轴标题
ax = plt.gca() # Get Current Axes 获取当前的子图
ax.grid(True) # 显示网格
plt.show()

