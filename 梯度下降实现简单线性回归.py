import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#  设定数据集
x_train=np.arange(0,15,1)
y_train=np.array([2,3,4,6,8,9,10,13,14,16,17,19,24,25,26])
#给定参数w,b
w=5
b=3

def compute_f_wb(x,w,b):    #计算y^=wx+b
    m=x.shape[0]     #获取长度
    f_wb=np.zeros(m)
    for i in range(m):
        f_wb[i]=w*x[i]+b
    return f_wb


def compute_cost(x,y,w,b):     #计算代价函数，min cost(w,b)即为最优值
    m=x.shape[0]
    cost=0
    for i in range(m):
        cost=cost+((w*x[i]+b)-y[i])**2
    cost=cost/2/m
    return cost

# print(compute_cost(x_train,y_train,w,b))

def compute_gradient(x,y,w,b):      #计算函数的梯度
    m=x.shape[0]
    dJ_dw=0          #将偏导数置零，为累加作准备
    dJ_db=0
    for i in range(m):
        f_wb=w*x[i]+b
        dJ_dw=dJ_dw+(f_wb-y[i])*x[i]
        dJ_db=dJ_db+(f_wb-y[i])
    dJ_dw=dJ_dw/m
    dJ_db=dJ_db/m
    return dJ_dw,dJ_db

def compute_linerRegression(x,y,w,b,num_iter,lr):    #梯度下降实现线性回归
    for i in range(num_iter):
        dJ_dw, dJ_db = compute_gradient(x, y, w, b)
        w=w-lr*dJ_dw
        b=b-lr*dJ_db
        if(compute_cost(x,y,w,b)==0):
            return w,b
    return w,b

#测试部分
w,b= compute_linerRegression(x_train, y_train, w, b, 100000, 0.01)#设定线性回归所需参数
f_wb=compute_f_wb(x_train,w,b)
print(f"w is {w},b is {b}\n")
print(f"cost is {compute_cost(x_train,y_train,w,b)}\n")
#测试部分

#创建画布
fig=plt.figure(figsize=plt.figaspect(2))
# fig,(ax1,ax2) = plt.subplots(1, 2)
# ax2.plot(x_train,y_train,c='b',label='feature y')

#第一个子图,绘制拟合的直线
ax1=fig.add_subplot(2,1,1)
ax1.scatter(x_train,y_train,marker='x',c='r')
ax1.plot(x_train,f_wb,c='b')
plt.xlabel('x')
plt.ylabel('y')


#第二个子图,绘制cost函数
ax=fig.add_subplot(2,1,2,projection='3d')
# ax = Axes3D(fig, auto_add_to_figure=False)
# fig.add_axes(ax)
w=np.arange(-5,5,0.1)
b=np.arange(-5,5,0.1)
X, Y = np.meshgrid(w, b)
Z =compute_cost(x_train,y_train,X,Y)
plt.xlabel('w')
plt.ylabel('b')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')


plt.show()






