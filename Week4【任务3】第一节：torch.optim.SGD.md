

Week4【任务3】第一节：torch.optim.SGD
$$
\begin{array}{l}
梯度下降：\boldsymbol{w}_{i+1}=\boldsymbol{w}_{i}-\boldsymbol{g}\left(\boldsymbol{w}_{i}\right)\\
y=f(x)=4 * x^{2} \\
y^{\prime}=f^{\prime}(x)=8 * x \\
x_{0}=2, \quad y_{0}=16, f^{\prime}\left(x_{0}\right)=16 \\
x_{1}=x_{0}-f^{\prime}\left(x_{0}\right)=2-16=-14 \\
x_{1}=-14, \quad y_{1}=784, f^{\prime}\left(x_{1}\right)=-112 \\
x_{2}=x_{6}-f^{\prime}\left(x_{1}\right)=-14+112=98, \quad y_{2}=38416
\end{array}
$$

#### lr=1

![image-20201231093415217](Week4%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Atorch.optim.SGD.assets/image-20201231093415217.png)

<img src="Week4%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Atorch.optim.SGD.assets/image-20201231093125911.png" alt="image-20201231093125911" style="zoom:50%;" />



### 乘以学习率

$$
w_{i+1}=w_{i}-L R \times \operatorname{grad}\left(w_{i}\right)
$$

#### lr=0.5

<img src="Week4%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Atorch.optim.SGD.assets/image-20201231100925937.png" alt="image-20201231100925937" style="zoom:50%;" />

<img src="Week4%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Atorch.optim.SGD.assets/image-20201231101003991.png" alt="image-20201231101003991" style="zoom:67%;" />



#### lr=0.2

<img src="Week4%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Atorch.optim.SGD.assets/image-20201231094855434.png" alt="image-20201231094855434" style="zoom:50%;" />

<img src="Week4%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Atorch.optim.SGD.assets/image-20201231094935494.png" alt="image-20201231094935494" style="zoom: 67%;" />

#### lr=0.1

![image-20201231093932643](Week4%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Atorch.optim.SGD.assets/image-20201231093932643.png)

#### lr=0.125 (上帝视角)

<img src="Week4%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Atorch.optim.SGD.assets/image-20201231094607365.png" alt="image-20201231094607365" style="zoom: 50%;" />

<img src="Week4%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Atorch.optim.SGD.assets/image-20201231095520709.png" alt="image-20201231095520709" style="zoom: 50%;" />

#### learning-rate

![image-20201231094150503](Week4%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Atorch.optim.SGD.assets/image-20201231094150503.png)

### multi learning rate

#### lr_max = 0.3

<img src="Week4%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Atorch.optim.SGD.assets/image-20201231094526886.png" alt="image-20201231094526886" style="zoom:67%;" />

![image-20201231094317984](Week4%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Atorch.optim.SGD.assets/image-20201231094317984.png)

<img src="Week4%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Atorch.optim.SGD.assets/image-20201231094328870.png" alt="image-20201231094328870" style="zoom:80%;" />

## Momentum

#### exponential weight

![图片](Week4%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Atorch.optim.SGD.assets/640)

指数加权平均， 指数加权平均在时间序列中经常用于求取平均值的一个方法，它的思想是：我们要求取当前时刻的平均值，距离当前时刻越近的那些参数值，它的参考性越大，所占的权重就越大，这个权重是随时间间隔的增大呈指数下降，所以叫做指数滑动平均。公式如下：
$$
v_{t}=\beta * v_{t-1}+(1-\beta) * \theta_{t}
$$
是当前时刻的一个平均值，这个平均值有两项构成，一项是当前时刻的参数值 $\theta_{t}$  , 所占的权重是$1-\beta$， 这个$\beta$ 是个参数。另一项是上一时刻的一个平均值，权重是 $\beta$。

![图片](https://mmbiz.qpic.cn/mmbiz_png/210IyDic7racmzhFcYJ3xGkoS1LtVdaKK5iaFzM6ppcq6seKe2DvGuP4BBNicPeQEx8mes7snhemmD1o2013URTFw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
$$
\begin{array}{l}
v_{100}=\beta * v_{99}+(1-\beta) * \theta_{100} \\
=(1-\beta) * \theta_{100}+\beta *\left(\beta * v_{98}+(1-\beta) * \theta_{99}\right) \\
=(1-\beta) * \theta_{100}+(1-\beta) * \beta * \theta_{99}+\left(\beta^{2} * v_{98}\right) \\
=(1-\beta) * \theta_{100}+(1-\beta) * \beta * \theta_{99}+(1-\beta) * \beta^{2} * \theta_{98}+\left(\beta^{3} * v_{97}\right) \\
=(1-\beta) * \beta^{0} * \theta_{100}+(1-\beta) * \beta^{1} * \theta_{99}+(1-\beta) * \beta^{2} * \theta_{98}+\left(\beta^{3} * v_{97}\right) \\
=\sum_{i}^{N}(\mathbf{1}-\boldsymbol{\beta}) * \boldsymbol{\beta}^{i} * \boldsymbol{\theta}_{N-i}
\end{array}
$$
![image-20201231102418439](Week4%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Atorch.optim.SGD.assets/image-20201231102418439.png)

<img src="Week4%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Atorch.optim.SGD.assets/image-20201231102450497.png" alt="image-20201231102450497" style="zoom:67%;" />

#### multi-weight

![image-20201231102553977](Week4%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Atorch.optim.SGD.assets/image-20201231102553977.png)

<img src="Week4%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Atorch.optim.SGD.assets/image-20201231102627384.png" alt="image-20201231102627384" style="zoom:67%;" />

### Optimizer

#### Momentum

$$
普通梯度下降:w_{i+1}=w_{i}-l r * g\left(w_{i}\right)
\\
momentum： 
v_{i}=m * v_{i-1}+g\left(w_{i}\right) w_{i+1}=w_{i}-l r * v_{i}
$$

$$
\begin{aligned}
v_{100} &=m * v_{99}+g\left(w_{100}\right) \\
&=g\left(w_{100}\right)+m *\left(m * v_{98}+g\left(w_{99}\right)\right) \\
&=g\left(w_{100}\right)+m * g\left(w_{99}\right)+m^{2} * v_{98} \\
&=g\left(w_{100}\right)+m * g\left(w_{99}\right)+m^{2} * g\left(w_{98}\right)+m^{3} * v_{97}
\end{aligned}
$$

$w_{i+1}$：第 $i+1$ 次更新的参数
$lr$ : 学习率
$v_{i}$: 更新量
$m$ : momentum系数
$g\left(w_{i}\right)$ : $w_{i}$的梯度

![image-20201231103536657](Week4%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Atorch.optim.SGD.assets/image-20201231103536657.png)

#### 都不加momentum

<img src="Week4%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Atorch.optim.SGD.assets/image-20201231115600337.png" alt="image-20201231115600337" style="zoom: 67%;" />

#### 给小学习率加一个momentum

<img src="Week4%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Atorch.optim.SGD.assets/image-20201231103548218.png" alt="image-20201231103548218" style="zoom:67%;" />

#### momentum = 0.63

![image-20201231103706201](Week4%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Atorch.optim.SGD.assets/image-20201231103706201.png)

### SGD

```python
optim.SGD(params, lr=<object object>, momentum=0, dampening=0, weight_decay=0, nesterov=False)

```

- param: 管理的参数组

- lr: 初识学习率

- momentum：动量系数， beta

- weight_decay: L2 正则化系数

- nesterov: 是否采用 NAG

  NAG参考文献:《On the importance of initialization and momentum in deep learning》

#### 10 款优化器：

- optim.SGD: 随机梯度下降法

- optim.Adagrad: 自适应学习率梯度下降法

- optim.RMSprop: Adagrad 的改进

- optim.Adadelta: Adagrad 的改进

- optim.Adam: RMSprop 结合 Momentum

- optim.Adamax: Adam 增加学习率上限

- optim.SparseAdam: 稀疏版的 Adam

- optim.ASGD: 随机平均梯度下降

- optim.Rprop: 弹性反向传播

- optim.LBFGS: BFGS 的改进

  

  #### 参考资料

1. optim.SGD :《On the importance of initialization and momentum in deep learning
2. optim.Adagrad :《Adaptive Subgradient Methods for Online Learning and Stochastic Optimization》
3. optim.RMSprop :
  http:.//www.cs.toronto.edu/-tijmen/csc321/slides/lecture_slides_lec6.pdf
4. optim.Adadelta : 《AN ADAPTIVE LEARNING RATE METHOD》
5. optim.Adam : 《Adam: A Method for Stochastic Optimization》
6. optim.Adamax : 《Adam: A Method for Stochastic Optimization》
7. optim.SparseAdam
8. optim.ASGD :《Accelerating Stochastic Gradient Descent using Predictiv Variance Reduction》
9. optim.Rprop :《Martin Riedmiller und Heinrich Braun》