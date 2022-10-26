# 这是一个示例 Python 脚本。

# 按 ⌃R 执行或将其替换为您的代码。
# 按 双击 ⇧ 在所有地方搜索类、文件、工具窗口、操作和设置。

import numpy as np
import matplotlib.pyplot as plt

#绘制房价散点图
x_train=np.arange(0,15,1)
y_train=np.array([400.0,700.0,1000.0,1200,1300,1700,1900,2100,2200,2400,2700,2900,2600,3000,3800])
plt.scatter(x_train,y_train,marker='x',c='r',label='actual prices')
# plt.plot(x_train,y_train,c='b',label='our predictions')
plt.title("Housing prices")
plt.ylabel("Price (in 1000s of dollars)")
plt.xlabel("Size (1000 sqft)")


def compute_f_wb(x,w,b):          #计算拟合曲线
    m=x.shape[0]        #获取长度
    f_wb=np.zeros(m)
    for i in range(m):
        f_wb[i]=w*x[i]+b
    return f_wb

w=500
b=200

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

'''测试部分'''
w,b= compute_linerRegression(x_train, y_train, w, b, 100000, 0.01)
f_wb=compute_f_wb(x_train,w,b)
print(f"w is {w},b is {b}\n")
print(f"cost is {compute_cost(x_train,y_train,w,b)}\n")
'''测试部分'''



'''绘制拟合直线'''
plt.plot(x_train,f_wb,c='b',label='our predictions')
plt.legend()
plt.show()
'''绘制拟合直线'''




