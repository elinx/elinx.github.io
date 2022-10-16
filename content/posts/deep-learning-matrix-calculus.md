+++
title = "Deep Learning Matrix Calculus"
date = "2022-10-12T22:28:01+08:00"
author = ""
authorTwitter = "" #do not include @
cover = ""
tags = ["Deep Learning", "Jacobian", "Gradient"]
keywords = ["", ""]
description = "All matrix knowledge you need to know in deep learning in a pratical way"
showFullContent = false
readingTime = false
hideComments = false
math = true
+++

# 预备知识
在下面的公式推导中，我们都用小写字母表示标量，比如$x, y, z$，用小写的加粗字母表示向量，比如$\bm{x, y, z}$，所有的公式中使用的向量都是列向量，比如长度为$\left|\bm{x}\right|=n$的向量为：


$$
\bm{x}=\begin{bmatrix} x_1 \\\\ x_2 \\\\ \vdots \\\\  x_n\end{bmatrix}
$$

逐点乘积(Hadamard Product, Element-wise Multiplication)：相同维度的向量或者矩阵每个元素的乘积，结果是一个向量，记为$\bm{w}\otimes\bm{x}$，表示的运算为：

$$
\bm{w}\otimes\bm{x}=\begin{bmatrix} w_1x_1 \\\\ w_2x_2 \\\\ \vdots \\\\  w_nx_n\end{bmatrix}
$$

点积(dot product)
我们用$\bm{w}\cdot\bm{x}$表示向量的点积，点积的结果是一个标量：
$$
\bm{w}\cdot\bm{x}=\sum_{i}^{n}{w_i*x_i}=sum(\bm{w}\otimes\bm{x})
$$

权重矩阵的表示方法：每个神经元的权重是一个列向量，多个神经元向量组成矩阵，我们遵守常用记号，将其作为神经元的行向量

矩阵的乘法(Matrix Multiplication)
矩阵的点乘()

Tensorflow和Pytorch中的Input矩阵在表示的时候都是将features放到了行上面，多个feature按照多行排布，比如下面表示特征向量有n个元素，有m个向量的$\bm{X}$其形状为$(M, N)$
$$
\bm{X} = \begin{bmatrix} \bm{x_1} \\\\ \bm{x_2} \\\\ \vdots \\\\  \bm{x_m}\end{bmatrix} = \begin{vmatrix}
x_{11} & x_{12} & \dots & x_{1n} \\\\
x_{21} & x_{22} & \dots & x_{2n} \\\\
\vdots & \vdots & \vdots & \vdots \\\\
x_{m1} & x_{m2} & \dots & x_{mn} \\\\
\end{vmatrix}
$$
对权重的放置却略有不同，Tensorflow会将权重向量放到列上，而Pytorch会放到行上，也就是对应于上面的输入向量，有K个神经元，每个神经元是N个特征的话，Tensorflow的表示的形状是$(N, K)$，也就是K个N维的列向量：
$$
\bm{W} = \begin{bmatrix} \bm{w_1}, \bm{w_2},\dots, \bm{w_k}\end{bmatrix} = \begin{vmatrix}
w_{11} & w_{21} & \dots & w_{k1} \\\\
w_{12} & w_{22} & \dots & w_{k2} \\\\
\vdots & \vdots & \vdots & \vdots \\\\
w_{1n} & w_{2n} & \dots & w_{kn} \\\\
\end{vmatrix}
$$
Pytorch的形状却是$(K, N)$，也就是K个N维的行向量：
$$
\bm{W} = \begin{bmatrix} \bm{w_1} \\\\ \bm{w_2} \\\\ \vdots \\\\ \bm{w_k}\end{bmatrix} = \begin{vmatrix}
w_{11} & w_{12} & \dots & w_{1n} \\\\
w_{21} & w_{22} & \dots & w_{2n} \\\\
\vdots & \vdots & \vdots & \vdots \\\\
w_{k1} & w_{k2} & \dots & w_{kn} \\\\
\end{vmatrix}
$$
因此其`Dense/Linear`的做法也不相同，Tensorflow是$\bm{y}=\bm{X}\bm{W}$，Pytorch的是$\bm{y}=\bm{X}\bm{W^T}$，得到的结果都是一样的，也就是batch是行，特征是列，有多少行就有多少个batch。

# 求导
## 标量对向量求导
对于函数$f(\bm{x})=y$, $y$对输入$\bm{x}$的导数跟输入$\bm{x}$的形状相同，也就是：
$$
\frac{dy}{d\bm{x}}=
  \begin{bmatrix}
    \frac{dy}{dx_1} \\\\
    \frac{dy}{dx_2} \\\\
    \vdots \\\\
    \frac{dy}{dx_n}
  \end{bmatrix}
$$
## 向量对向量求导
不同于上面的$\bm{y}$是一个标量，这里的$\bm{y}$是一个m维的向量，对应的表达式是$\bm{y}=\bm{f}(\bm{x})$展开之后是：
$$
\begin{bmatrix}
  y_1 \\\\ y_2 \\\\ \vdots \\\\ y_m
\end{bmatrix}=
\begin{bmatrix}
  f_1(\bm{x}) \\\\
  f_2(\bm{x}) \\\\
  \vdots \\\\
  f_m(\bm{x}) \\\\
\end{bmatrix}=
\begin{bmatrix}
  f_1(x_1, x_2, ..., x_n) \\\\
  f_2(x_1, x_2, ..., x_n) \\\\
  \vdots \\\\
  f_n(x_1, x_2, ..., x_n) \\\\
\end{bmatrix}
$$

而$\bm{y}$对$\bm{x}$的倒数也就成了对$\bm{x}$各个分量的偏导数，也就是有：

$$
\begin{gather*}
  \displaystyle\frac{\partial{\bm{y}}}{\partial{\bm{x}}}=
  \left[
    \begin{array}{cccc}
      \displaystyle\frac{\partial{f_1}}{\partial{\bm{x}}}\\\\
      \displaystyle\frac{\partial{f_2}}{\partial{\bm{x}}}\\\\
      \vdots\\\\
      \displaystyle\frac{\partial{f_m}}{\partial{\bm{x}}}
    \end{array}
  \right]=
  \left[
    \begin{array}{cccc}
      \displaystyle\frac{\partial{f_1}}{\partial{x_1}}&
      \displaystyle\frac{\partial{f_1}}{\partial{x_2}}&
      \ldots &
      \displaystyle\frac{\partial{f_1}}{\partial{x_n}}\\\\
      \displaystyle\frac{\partial{f_2}}{\partial{x_1}}&
      \displaystyle\frac{\partial{f_2}}{\partial{x_2}}&
      \ldots &
      \displaystyle\frac{\partial{f_2}}{\partial{x_n}}\\\\
      \vdots & \vdots & \ddots & \vdots \\\\
      \displaystyle\frac{\partial{f_m}}{\partial{x_1}}&
      \displaystyle\frac{\partial{f_m}}{\partial{x_2}}&
      \ldots &
      \displaystyle\frac{\partial{f_m}}{\partial{x_n}}
    \end{array}
  \right]
\end{gather*}
$$

### 逐元素(Element-wise)操作的求导
这种比较常出现在激活函数，也就是对输入向量的每个元素求值，更泛化的形式是两个矩阵逐个进行binary操作，我们将其记为$\bm{y}=\bm{f}(\bm{w})\circ\bm{g}(\bm{x})$，这里的$\bm{y}, \bm{f}, \bm{g}, \bm{w},\bm{x}$都是向量并且其长度都是n，也就有：
$$
\begin{bmatrix}
  y_1 \\\\ y_2 \\\\ \vdots \\\\ y_n
\end{bmatrix}=
\begin{bmatrix}
  f_1(\bm{w}) \circ g_1(\bm{x}) \\\\
  f_2(\bm{w}) \circ g_2(\bm{x}) \\\\
  \vdots \\\\
  f_n(\bm{w}) \circ g_n(\bm{x}) \\\\
\end{bmatrix}=
\begin{bmatrix}
  f_1(w_1, w_2, ..., w_n) \circ g_1(x_1, x_2, ..., x_n) \\\\
  f_2(w_1, w_2, ..., w_n) \circ g_2(x_1, x_2, ..., x_n) \\\\
  \vdots \\\\
  f_n(w_1, w_2, ..., w_n) \circ g_n(x_1, x_2, ..., x_n) \\\\
\end{bmatrix}
$$
套用上面求导的一般形式我们得到一般形式为：
$$
\begin{gather*}
  \displaystyle\frac{\partial{\bm{y}}}{\partial{\bm{x}}}=
  \left[
    \begin{array}{cccc}
      \displaystyle\frac{\partial{(f_1\circ g_1)}}{\partial{\bm{x}}}\\\\
      \displaystyle\frac{\partial{(f_2\circ g_2)}}{\partial{\bm{x}}}\\\\
      \vdots\\\\
      \displaystyle\frac{\partial{(f_n\circ g_n)}}{\partial{\bm{x}}}
    \end{array}
  \right]=
  \left[
    \begin{array}{cccc}
      \displaystyle\frac{\partial{(f_1\circ g_1)}}{\partial{x_1}}&
      \displaystyle\frac{\partial{(f_1\circ g_1)}}{\partial{x_2}}&
      \ldots &
      \displaystyle\frac{\partial{(f_1\circ g_1)}}{\partial{x_n}}\\\\
      \displaystyle\frac{\partial{(f_2\circ g_2)}}{\partial{x_1}}&
      \displaystyle\frac{\partial{(f_2\circ g_2)}}{\partial{x_2}}&
      \ldots &
      \displaystyle\frac{\partial{(f_2\circ g_2)}}{\partial{x_n}}\\\\
      \vdots & \vdots & \ddots & \vdots \\\\
      \displaystyle\frac{\partial{(f_n\circ g_n)}}{\partial{x_1}}&
      \displaystyle\frac{\partial{(f_n\circ g_n)}}{\partial{x_2}}&
      \ldots &
      \displaystyle\frac{\partial{(f_n\circ g_n)}}{\partial{x_n}}
    \end{array}
  \right]
\end{gather*}
$$

在深度学习中我们的操作如果是参数乘以输入的话呢，$\bm{f, g}$就都是相当于一个选择函数，即$f_i(\bm{w})=w_i$和$g_i(\bm{x})=x_i$，这样面的矩阵就会变成一个对角阵
$$
\begin{bmatrix}
  y_1 \\\\ y_2 \\\\ \vdots \\\\ y_n
\end{bmatrix}=
\begin{bmatrix}
  f_1(\bm{w}) \circ g_1(\bm{x}) \\\\
  f_2(\bm{w}) \circ g_2(\bm{x}) \\\\
  \vdots \\\\
  f_n(\bm{w}) \circ g_n(\bm{x}) \\\\
\end{bmatrix}=
\begin{bmatrix}
  w_1 \circ x_1 \\\\
  w_2 \circ x_2 \\\\
  \vdots \\\\
  w_n \circ x_n \\\\
\end{bmatrix}
$$
也就是说$y_1$只跟$x_1$有关，$y_2$只跟$x_2$有关，$y_n$只跟$x_n$有关，其他位置的元素都无关，也就是：
$$
\begin{gather*}
  \displaystyle\frac{\partial{\bm{y}}}{\partial{\bm{x}}}=
  \left[
    \begin{array}{cccc}
      \displaystyle\frac{\partial{(w_1\circ x_1)}}{\partial{\bm{x}}}\\\\
      \displaystyle\frac{\partial{(w_2\circ x_2)}}{\partial{\bm{x}}}\\\\
      \vdots\\\\
      \displaystyle\frac{\partial{(w_n\circ x_n)}}{\partial{\bm{x}}}
    \end{array}
  \right]=
  \left[
    \begin{array}{cccc}
      \displaystyle\frac{\partial{(w_1\circ x_1)}}{\partial{x_1}}&
      \displaystyle 0&
      \ldots &
      \displaystyle 0\\\\
      \displaystyle 0&
      \displaystyle\frac{\partial{(w_2\circ x_2)}}{\partial{x_2}}&
      \ldots &
      \displaystyle 0\\\\
      \vdots & \vdots & \ddots & \vdots \\\\
      \displaystyle 0&
      \displaystyle 0&
      \ldots &
      \displaystyle\frac{\partial{(w_n\circ x_n)}}{\partial{x_n}}
    \end{array}
  \right]
\end{gather*}
$$
如果这里是乘法的话就变成了参数的对角矩阵，同样的对参数的导数是关于输入向量的导数。我们用代码来验证一下结果是否正确，我们调用Pytorch的`jacobian`来计算一下，首先我们需要引入一些必要的基础库，后面我们不再列出：
```python
import torch
from torch.autograd.functional import jacobian
from torch import tensor
```
我们计算一个element-wise的矩阵乘法并分别对两个向量求导，可以看到结果跟我们上面的公式一致。
```python
x = tensor([1., 2, 3])
w = tensor([4., 5, 6])

y = w * x
dx = jacobian(lambda x: w * x, x)
dw = jacobian(lambda w: w * x, w)
print(f'y: {y},\n\ndy/dx: {dx},\n\ndy/dw: {dw}')

# output
# y: tensor([ 4., 10., 18.]),
# 
# dy/dx: tensor([[4., 0., 0.],
#         [0., 5., 0.],
#         [0., 0., 6.]]),
# 
# dy/dw: tensor([[1., 0., 0.],
#         [0., 2., 0.],
#         [0., 0., 3.]]
```

### 点积(dot product)的求导
向量的点积表示为$\bm{w}\cdot\bm{x}$，也可以表示为$\bm{w^T}\bm{x}$，同时还可以看作是element-wise向量乘操作之后再执行一个加和操作，也就是$\sum_i^n{w_ix_i}=sum(\bm{w}\otimes\bm{x})$，借用这个转换我们可以利用链式法则求出向量点积的导数。

我们先定义一个中间变量$\bm{u}$将表达式重写
$$
\bm{u}=\bm{w}\otimes\bm{x} \\\\
y=sum(\bm{u})
$$
然后求各个中间变量对于$\bm{w}$的导数
$$
\frac{\partial{u}}{\partial{w}}=diag(\bm{x})= 
\begin{bmatrix}
  x_1&0&\dots&0\\\\
  0&x_2&\dots&0\\\\
  \vdots&\vdots&\ddots&\vdots\\\\
  0&0&\dots&x_n
\end{bmatrix} \\\\
\frac{\partial{y}}{\partial{u}}=\vec{1}^T = [1_1, 1_2, ..., 1_n]
$$

两个中间结果的乘积就是对$\bm{w}$的导数
$$
\frac{\partial{y}}{\partial{\bm{w}}}=
  \frac{\partial{y}}{\partial{\bm{u}}}\frac{\partial{\bm{u}}}{\partial{\bm{w}}}=
  \vec{1}^Tdiag(\bm{x})=[x_1, x_2, \dots, x_n]=\bm{x^T}
$$

同样的如果我们得到对$\bm{x}$的导数
$$
\frac{\partial{y}}{\partial{\bm{x}}}=\bm{w^T}
$$
TODO: 上面求导的结果跟变量的形状不匹配，是不是定义有问题？

同样用代码验证一下
```python
x = tensor([1., 2, 3])
w = tensor([4., 5, 6])

y = torch.dot(w, x)
dx = jacobian(lambda x: torch.dot(w, x), x)
dw = jacobian(lambda w: torch.dot(w, x), w)
print(f'y: {y},\n\ndy/dx: {dx},\n\ndy/dw: {dw}')
# y: 32.0,
# 
# dy/dx: tensor([4., 5., 6.]),
# 
# dy/dw: tensor([1., 2., 3.])
```
可以看到`torch.dot`跟我们推出来的公式跟这里并不一致，形状有点差别，导数的形状跟输入的另一个参数形状相同了，为什么呢？
TODO: 看起来像是特殊处理的。
```python
x = tensor([[1.], [2.], [3.]])
w = tensor([4., 5, 6])

y = torch.matmul(w, x)
dx = jacobian(lambda x: torch.matmul(w, x), x)
dw = jacobian(lambda w: torch.matmul(w, x), w)
print(f'y: {y},\n\ndy/dx: {dx},\n\ndy/dw: {dw}')
print(dx.shape, dw.shape)
# y: tensor([32.]),
# 
# dy/dx: tensor([[[4.],
#          [5.],
#          [6.]]]),
# 
# dy/dw: tensor([[1., 2., 3.]])
# torch.Size([1, 3, 1]) torch.Size([1, 3])
```
但是看`torch.matmul`就跟我们求出来的基本一致，但是这里`y`是一个1x1矩阵，`dx`的形状也多了一维，但是仍然可以进行减法计算，而`dw`的形状是对的。

`torch.matmul`还有一个性质就是如果输入都是一维向量的话结果是求两个的点积，也就是跟点积完全一样了，所以应该`torch.dot`底层也是用`torch.matmul`实现的。
```python
x = tensor([1., 2, 3])
w = tensor([4., 5, 6])

y = torch.matmul(w, x)
dx = jacobian(lambda x: torch.matmul(w, x), x)
dw = jacobian(lambda w: torch.matmul(w, x), w)
print(f'y: {y},\n\ndy/dx: {dx},\n\ndy/dw: {dw}')
# y: 32.0,
# 
# dy/dx: tensor([4., 5., 6.]),
# 
# dy/dw: tensor([1., 2., 3.])
```

## 向量对矩阵求导
向量和矩阵的乘积是一个向量，怎么求向量对矩阵的导数呢？我们考察一下batch个input向量的情形，也就是假设一个神经元$\bm{w}$，m张图片输入记做$\bm{X}$，其形状是$(M, N)$：
$$
X=
  \begin{bmatrix}
    \bm{x_1},\bm{x_2},\dots,\bm{x_m}
  \end{bmatrix}^T
$$
## 矩阵对矩阵求导