

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20201215205531516.png" alt="image-20201215205531516" style="zoom:50%;" />

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20201215205534737.png" alt="image-20201215205534737" style="zoom: 67%;" />



### nn.MaxPoo12d
功能:对二维信号（图像)进行最大值池化主要参数:

<img src="Week3%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%BD%91%E7%BB%9C%E5%B1%82%E4%B8%AD%E7%9A%84%E6%B1%A0%E5%8C%96%E5%B1%82%EF%BC%8C%E5%85%A8%E8%BF%9E%E6%8E%A5%E5%B1%82%E5%92%8C%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%E5%B1%82.assets/image-20201215205853901.png" alt="image-20201215205853901" style="zoom: 67%;" />

kernel size:池化核尺寸  stride:步长
padding :填充个数  dilation: 池化核间隔大小  ceil_mode: 尺寸向上取整
return indices:记录池化像素索引

### nn.AvgPoo12d
功能:对二维信号（图像）进行平均值池化主要参数:

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20201216114224776.png" alt="image-20201216114224776" style="zoom:67%;" />

kernel_size:池化核尺寸  stride:步长
padding :填充个数
ceil_mode:尺寸向上取整
count_include_pad:填充值用于计算
divisor_override :除法因子

<img src="Week3%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%BD%91%E7%BB%9C%E5%B1%82%E4%B8%AD%E7%9A%84%E6%B1%A0%E5%8C%96%E5%B1%82%EF%BC%8C%E5%85%A8%E8%BF%9E%E6%8E%A5%E5%B1%82%E5%92%8C%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%E5%B1%82.assets/image-20201216120818967.png" alt="image-20201216120818967" style="zoom: 50%;" />

### nn.MaxUnpool2d

<img src="Week3%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%BD%91%E7%BB%9C%E5%B1%82%E4%B8%AD%E7%9A%84%E6%B1%A0%E5%8C%96%E5%B1%82%EF%BC%8C%E5%85%A8%E8%BF%9E%E6%8E%A5%E5%B1%82%E5%92%8C%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%E5%B1%82.assets/image-20201216113919041.png" alt="image-20201216113919041" style="zoom: 67%;" />

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20201216123008572.png" alt="image-20201216123008572" style="zoom: 50%;" />

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20201215205625776.png" alt="image-20201215205625776" />





### nn.Linear
功能:对一维信号(向量)进行线性组合

<img src="Week3%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%BD%91%E7%BB%9C%E5%B1%82%E4%B8%AD%E7%9A%84%E6%B1%A0%E5%8C%96%E5%B1%82%EF%BC%8C%E5%85%A8%E8%BF%9E%E6%8E%A5%E5%B1%82%E5%92%8C%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%E5%B1%82.assets/image-20201216102053124.png" alt="image-20201216102053124" style="zoom: 33%;" />

<img src="Week3%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%BD%91%E7%BB%9C%E5%B1%82%E4%B8%AD%E7%9A%84%E6%B1%A0%E5%8C%96%E5%B1%82%EF%BC%8C%E5%85%A8%E8%BF%9E%E6%8E%A5%E5%B1%82%E5%92%8C%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%E5%B1%82.assets/image-20201216102154868.png" alt="image-20201216102154868" style="zoom:50%;" />

主要参数:
in_features:输入结点数
out_features:输出结点数

bias :是否需要偏置
$$
\mathrm{y}=x W^{T}+\text { bias }
$$

```python
nn.Linear(in_features , out_features , bias=True)
```

<img src="Week3%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%BD%91%E7%BB%9C%E5%B1%82%E4%B8%AD%E7%9A%84%E6%B1%A0%E5%8C%96%E5%B1%82%EF%BC%8C%E5%85%A8%E8%BF%9E%E6%8E%A5%E5%B1%82%E5%92%8C%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%E5%B1%82.assets/image-20201216103009371.png" alt="image-20201216103009371" style="zoom:67%;" />
$$
\begin{array}{rl}
\boldsymbol{H}_{\mathbf{1}}=\boldsymbol{X} & * \boldsymbol{W}_{\mathbf{1}} \\
\boldsymbol{H}_{\mathbf{2}}=\boldsymbol{H}_{\mathbf{1}} & * \boldsymbol{W}_{\mathbf{2}} \\
\text { 0ut } \boldsymbol{p} u \boldsymbol{t} & =\boldsymbol{H}_{2} * \boldsymbol{W}_{3} \\
& =\boldsymbol{H}_{\mathbf{1}} * \boldsymbol{W}_{2} * \boldsymbol{W}_{3} \\
& =\boldsymbol{X} *\left(\boldsymbol{W}_{\mathbf{1}} * \boldsymbol{W}_{2} * \boldsymbol{W}_{3}\right) \\
& =\boldsymbol{X} * \boldsymbol{W}
\end{array}
$$
![image-20201216105003224](Week3%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%BD%91%E7%BB%9C%E5%B1%82%E4%B8%AD%E7%9A%84%E6%B1%A0%E5%8C%96%E5%B1%82%EF%BC%8C%E5%85%A8%E8%BF%9E%E6%8E%A5%E5%B1%82%E5%92%8C%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%E5%B1%82.assets/image-20201216105003224.png)

![image-20201216105023809](Week3%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%BD%91%E7%BB%9C%E5%B1%82%E4%B8%AD%E7%9A%84%E6%B1%A0%E5%8C%96%E5%B1%82%EF%BC%8C%E5%85%A8%E8%BF%9E%E6%8E%A5%E5%B1%82%E5%92%8C%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%E5%B1%82.assets/image-20201216105023809.png)

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20201216123413412.png" alt="image-20201216123413412" style="zoom:67%;" />



<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20201216123413412.png" alt="image-20201216123413412" style="zoom:67%;" />

### Activation function

#### Sigmoid

$$
\mathbf{y}=\frac{1}{1+e^{-x}}
\\
y^{\prime}=y *(1-y)
\
$$

特性:
输出值在（0,1)，符合概率
导数范围是[0，0.25]，易导致梯度消失
输出为非O均值,破坏数据分布

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20201216125513582.png" alt="image-20201216125513582" style="zoom: 50%;" />

### Tanh

$$
\begin{aligned}
&\text { 计算公式： } \mathbf{y}=\frac{\sin x}{\cos x}=\frac{e^{x}-e^{-x}}{e^{-}+e^{-x}}=\frac{2}{1+e^{-2 x}}+1\\
&\text { 梯度公式： } y^{\prime}=1-y^{2}
\end{aligned}
$$

特性:
输出值在(-1,1)，数据符合0均值

导数范围是(0.1)，易导致梯度消失



<img src="Week3%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%BD%91%E7%BB%9C%E5%B1%82%E4%B8%AD%E7%9A%84%E6%B1%A0%E5%8C%96%E5%B1%82%EF%BC%8C%E5%85%A8%E8%BF%9E%E6%8E%A5%E5%B1%82%E5%92%8C%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%E5%B1%82.assets/image-20201216130056318.png" alt="image-20201216130056318" style="zoom:50%;" />



#### ReLU

$$
\begin{aligned}
&\text { nn.ReLU }\\
&\text { 计算公式： } \mathbf{y}=\max (\mathbf{0}, \boldsymbol{x})\\
&\text { 梯度公式： } y^{\prime}=\left\{\begin{array}{ll}
1, & x>0 \\
u n d e f i n e d, & x=0 \\
0, & x<0
\end{array}\right.
\end{aligned}
$$



<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20201216130312367.png" alt="image-20201216130312367" style="zoom:50%;" />

特性:

输出值均为正数，负半轴导致死神经元

导数是1，缓解梯度消失,但易引发梯度爆炸

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20201216130312367.png" alt="image-20201216130312367" style="zoom: 50%;" />

#### nn.LeakyReLU
· negative_slope: 负半轴斜率
#### nn.PReLU
· init:可学习斜率
#### nn.RReLU
· lower:均匀分布下限
· upper:均匀分布上限

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20201216130605246.png" alt="image-20201216130605246" style="zoom:50%;" />