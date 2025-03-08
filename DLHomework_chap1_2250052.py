import numpy as np
import matplotlib.pyplot as plt

# 1. 建立一维数组 a
a = np.array([4, 5, 6])
print("a 的类型:", type(a))  # (1) 输出类型
print("a 的 shape:", a.shape)  # (2) 输出各维度的大小
print("a 的第一个元素:", a[0])  # (3) 输出第一个元素

# 2. 建立二维数组 b
b = np.array([[4, 5, 6], [1, 2, 3]])
print("b 的 shape:", b.shape)  # (1) 输出各维度的大小
print("b(0,0):", b[0, 0], "b(0,1):", b[0, 1], "b(1,1):", b[1, 1])  # (2) 输出指定元素

# 3. 创建特定矩阵
zero_matrix = np.zeros((3, 3), dtype=int)  # (1) 3x3全0矩阵
one_matrix = np.ones((4, 5))  # (2) 4x5全1矩阵
identity_matrix = np.eye(4)  # (3) 4x4单位矩阵
random_matrix = np.random.rand(3, 2)  # (4) 3x2随机矩阵

# 4. 建立数组 a
array_a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print("array_a:\n", array_a)  # (1) 打印 a
print("(2,3):", array_a[2, 3], "(0,0):", array_a[0, 0])  # (2) 输出指定元素

# 5. 选取 a 的 0到1行 2到3列 并存入 b
b = array_a[0:2, 2:4]
print("b:\n", b)  # (1) 输出 b
print("b(0,0):", b[0, 0])  # (2) 输出 b 的 (0,0) 元素

# 6. 选取 a 的最后两行所有元素存入 c
c = array_a[1:, :]
print("c:\n", c)  # (1) 输出 c
print("c 第一行最后一个元素:", c[0, -1])  # (2) 输出 c 第一行的最后一个元素

# 7. 特定索引访问数组 a
special_a = np.array([[1, 2], [3, 4], [5, 6]])
print("特殊索引元素:", special_a[[0, 1, 2], [0, 1, 0]])

# 8. 建立矩阵 a 并使用索引数组访问
matrix_a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
b_indices = np.array([0, 2, 0, 1])
print("特定索引访问结果:", matrix_a[np.arange(4), b_indices])

# 9. 每个选定元素加10
matrix_a[np.arange(4), b_indices] += 10
print("修改后的 matrix_a:\n", matrix_a)

# 10. 数据类型测试
x = np.array([1, 2])
print("x 的数据类型:", x.dtype)

x = np.array([1.0, 2.0])
print("x 的数据类型:", x.dtype)

# 11. 数学运算
x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)
print("x+y:\n", x + y)
print("np.add(x, y):\n", np.add(x, y))
print("x-y:\n", x - y)
print("np.subtract(x, y):\n", np.subtract(x, y))
print("x*y:\n", x * y)
print("np.multiply(x, y):\n", np.multiply(x, y))
print("np.dot(x, y):\n", np.dot(x, y))
print("x / y:\n", np.divide(x, y))
print("x 的开方:\n", np.sqrt(x))

# 12. 点积
print("x.dot(y):\n", x.dot(y))
print("np.dot(x, y):\n", np.dot(x, y))

# 13. 求和与平均值
print("np.sum(x):", np.sum(x))
print("np.sum(x, axis=0):", np.sum(x, axis=0))
print("np.sum(x, axis=1):", np.sum(x, axis=1))
print("np.mean(x):", np.mean(x))
print("np.mean(x, axis=0):", np.mean(x, axis=0))
print("np.mean(x, axis=1):", np.mean(x, axis=1))

# 14. 矩阵转置
print("x.T:\n", x.T)

# 15. 指数运算
print("np.exp(x):\n", np.exp(x))

# 16. 最大值索引
print("np.argmax(x):", np.argmax(x))
print("np.argmax(x, axis=0):", np.argmax(x, axis=0))
print("np.argmax(x, axis=1):", np.argmax(x, axis=1))

# 17. 画图 y = x^2
x_vals = np.arange(0, 100, 0.1)
y_vals = x_vals ** 2
plt.plot(x_vals, y_vals)
plt.title("y = x^2")
plt.show()

# 18. 画正弦和余弦函数
x_vals = np.arange(0, 3 * np.pi, 0.1)
sin_vals = np.sin(x_vals)
cos_vals = np.cos(x_vals)
plt.plot(x_vals, sin_vals, label="sin(x)")
plt.plot(x_vals, cos_vals, label="cos(x)")
plt.legend()
plt.title("Sine and Cosine Functions")
plt.show()
