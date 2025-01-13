import numpy as np
import matplotlib.pyplot as plt

def fit(y_data, degree=3):
    # 使用 polyfit 拟合数据
    
    coefficients = np.polyfit(x_data, y_data, degree)  # 使用最小二乘法拟合多项式

    # 生成拟合的多项式函数
    poly_func = np.poly1d(coefficients)

    # 预测新值
    y_fit = poly_func(x_data)

    # 绘制原始数据和拟合结果
    plt.scatter(x_data, y_data, label='Original Data', color='blue', alpha=0.6)
    plt.plot(x_data, y_fit, label=f'{degree}-degree Polynomial Fit', color='red', linewidth=2)
    plt.legend()
    plt.savefig('fit_NF1.png')

    # 输出拟合的多项式系数
    print("Fitted Polynomial Coefficients:")
    print(coefficients)

    print("\nFitted Polynomial:")
    print(poly_func)

    print("\nerror is: {}".format(np.max(abs(y_data - y_fit))))

# f1 的拟合 
x_data = np.linspace(-1.57, 1.57, 100)  # x 轴的 100 个数据点
y_data = np.exp(x_data) - 1
fit(y_data)

# f2 的拟合 
x_data = np.linspace(-1.57, 1.57, 100)  # x 轴的 100 个数据点
y_data = -(np.sin(x_data)) ** 2  # 非多项式函数加上噪声
fit(y_data, degree=4)

'''
Fitted Polynomial Coefficients:
[ 0.19130961  0.59629368  0.98648035 -0.02475941]

Fitted Polynomial:
        3          2
0.1913 x + 0.5963 x + 0.9865 x - 0.02476

error is: 0.07248146238312758
Fitted Polynomial Coefficients:
[ 9.77688475e-16 -4.54558068e-01 -1.84319678e-15 -1.23723894e-01]

Fitted Polynomial:
           3          2
9.777e-16 x - 0.4546 x - 1.843e-15 x - 0.1237

error is: 0.24416471067351453
'''

'''
Fitted Polynomial Coefficients:
[ 0.19130961  0.59629368  0.98648035 -0.02475941]

Fitted Polynomial:
        3          2
0.1913 x + 0.5963 x + 0.9865 x - 0.02476

error is: 0.07248146238312758
Fitted Polynomial Coefficients:
[ 2.07437811e-01 -1.17879390e-16 -9.01531606e-01  5.37437468e-17
 -1.13761155e-02]

Fitted Polynomial:
        4             3          2
0.2074 x - 1.179e-16 x - 0.9015 x + 5.374e-17 x - 0.01138

error is: 0.02677454477719443
'''