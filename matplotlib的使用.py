import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3,3,50)
y1 = 0.1*x
y2 = x**2

plt.figure(figsize=(10,5))
l1, = plt.plot(x,y2,label='linear line',zorder=2)
l2, = plt.plot(x,y1,color='red',linewidth=10,label='square line',zorder=1)

# .legend 给图像添加图例，显示的信息来自上面的 label
plt.legend(loc='best')
# 添加图例的位置参数有多种：什么都不写时，默认左上角，best（自动分配最佳位置）、（upper right|center|left）、（lower left|center|right）、（center left|right、center）、right（和 center right 位置同）

# plt.xlim 设置 x坐标轴 的范围
plt.xlim((-1,2))
plt.ylim((-2,3))
plt.xlabel('I am x')
plt.ylabel('I am y')
# np.linspace 定义范围及个数
new_xticks = np.linspace(-1,2,6)
# plt.xticks 设置 x轴 刻度，范围是 -1到2，刻度数为 6
plt.xticks(new_xticks)
plt.yticks([-2, -1.8, -1,0, 1.22, 3],[r'$really\ bad$',r'$bad$',r'$normal$',r'$zero$',r'$good$',r'$really\ good$']) # 改变y轴刻度的名称

ax = plt.gca() # 获取当前坐标轴信息

ax.spines['right'].set_color('none') # .spines 设置边框  .set_color 设置边框颜色
ax.spines['top'].set_color('none') # none 不显示上边框和右边框

ax.xaxis.set_ticks_position('bottom') # .xaxis.set_ticks_position 设置 x坐标刻度数字在边框的哪里 top（刻度和数字均在上边框）bottom（刻度和数字均在下边框）both（上下边框均有刻度，数字在下边框）default（同both）none（上下边框均无刻度，下边框有数字）
ax.yaxis.set_ticks_position('left') # .yaxis.set_ticks_position 设置 y轴坐标刻度位置，可选项 left\right\both\default\none

ax.spines['bottom'].set_position(('data',0)) # .spines['bottom']指的就是x轴，x轴相对与y轴的位置，在y=0处   （data\outward\axes）（outward 为0时，x轴是在y轴最底部，为正值时，x轴在 y轴底部下方 一定距离；为负值时，x轴在 y轴底部上方 一定距离）（axes 为0时，x轴在y轴底部位置；为1时，x轴在y轴顶部位置；为其他值时，x轴消失）（data 的话，指的是y轴的刻度，data是几，x轴就在y=几处）
ax.spines['left'].set_position(('data',0)) # .spines['let']指的就是y轴，y轴在 x=0 处

# 刻度的能见度 bbox 刻度背后的背景框
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(12)
    label.set_bbox(dict(facecolor='white',edgecolor='None',alpha=0.7,zorder=3))

# annotation 标注
x0 = 1
y0 = x0**2
plt.plot([x0,x0],[0,y0],'k--',linewidth=2.5) # (x0,y0) 垂直到 x轴 的虚线；'k--' 黑色虚线
plt.scatter([x0,],[y0,],s=50,color='b') # (x0,y0)位置画一个点
# 添加注释 annotate
plt.annotate(r'$x^2=%s$' % y0,xy=(x0,y0),xycoords='data',xytext=(+30,-30),textcoords='offset points',fontsize=16,arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2'))
# 添加注释 text
plt.text(0.2,-1,r'$This\ is\ the\ some\ text.\mu\ \delta_i\ \alpha_t$',fontdict={'size':16,'color':'red'})
plt.show()

#######################
# 散点图
n = 1024
# np.random.randn(size) 标准正态分布（μ=0, σ=1）
# 对应于 np.random.normal(loc=0,scale=1,size)
X = np.random.randn(n)
Y = np.random.normal(0,1,n)
T = np.arctan2(Y,X)
plt.scatter(X,Y,s=50,c=T,alpha=0.5,cmap=plt.cm.Paired)
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.show()