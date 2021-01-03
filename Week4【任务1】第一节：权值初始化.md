$\begin{aligned} \mathrm{H}_{2}=& \mathrm{H}_{1} * \mathrm{~W}_{2} \\ \Delta \mathrm{W}_{2} &=\frac{\partial \mathrm{Loss}}{\partial \mathrm{W}_{2}}=\frac{\partial \mathrm{Loss}}{\partial \text { out }} * \frac{\partial \text { out }}{\partial \mathrm{H}_{2}} \star \frac{\partial \mathrm{H}_{2}}{\partial \mathrm{w}_{2}} \\ &=\frac{\partial \mathrm{Loss}}{\partial \text { out }} \star \frac{\partial \text { out }}{\partial \mathrm{H}_{2}} * \mathrm{H}_{1} \end{aligned}$
$$
\begin{aligned}&\text { 梯度消失 }: \mathrm{H}_{1} \rightarrow \mathbf{0} \Rightarrow \Delta \mathrm{W}_{2} \rightarrow \mathbf{0}\\&\text { 梯度爆炸 }: \mathrm{H}_{1} \rightarrow \infty \quad \Rightarrow \Delta \mathrm{W}_{2} \rightarrow \infty\end{aligned}
$$

<img src="Week4%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E6%9D%83%E5%80%BC%E5%88%9D%E5%A7%8B%E5%8C%96.assets/image-20201216162122990.png" alt="image-20201216162122990" style="zoom:80%;" />

<img src="Week4%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E6%9D%83%E5%80%BC%E5%88%9D%E5%A7%8B%E5%8C%96.assets/image-20201216162225084.png" alt="image-20201216162225084" style="zoom:80%;" />



#### Gradient Vanishing and Exploding

1. $\mathrm{E}(\boldsymbol{X} * \boldsymbol{Y})=\boldsymbol{E}(\boldsymbol{X}) * \boldsymbol{E}(\boldsymbol{Y})$
2. $\mathrm{D}(\boldsymbol{X})=\boldsymbol{E}\left(\mathrm{X}^{2}\right)-[\boldsymbol{E}(\boldsymbol{X})]^{2}$
3. $\mathrm{D}(\boldsymbol{X}+\boldsymbol{Y})=\boldsymbol{D}(\boldsymbol{X})+\boldsymbol{D}(\boldsymbol{Y})$

$1.2 .3 \Rightarrow \mathrm{D}(\mathrm{X} * \mathrm{Y})=\mathrm{D}(\mathrm{X}) * \mathrm{D}(\mathrm{Y})+\mathrm{D}(\mathrm{X}) *[\boldsymbol{E}(\boldsymbol{Y})]^{2}+\mathrm{D}(\mathrm{Y})^{*}[\boldsymbol{E}(\boldsymbol{X})]^{2}$

X, Y 均值为0，期望为0

若 $E(X)=0, E(Y)=0$
$D(X * Y)=D(X) * D(Y)$

$\mathrm{H}_{11}=\sum_{i=0}^{n} X_{i} * W_{1 i} \quad \mathrm{D}(\mathrm{X} * \mathrm{Y})=\mathrm{D}(\mathrm{X}) * \mathrm{D}(\mathrm{Y})$
$\mathrm{D}\left(\mathrm{H}_{11}\right)=\sum_{i=0}^{n} D\left(X_{i}\right) * D\left(W_{1 i}\right)$
$\quad=\mathrm{n} *(1 * 1)$
$\quad=\mathrm{n}$
$\operatorname{std}\left(\mathrm{H}_{11}\right)=\sqrt{\mathrm{D}\left(\mathrm{H}_{11}\right)}=\sqrt{n}$



<img src="Week4%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E6%9D%83%E5%80%BC%E5%88%9D%E5%A7%8B%E5%8C%96.assets/image-20201216162406940.png" alt="image-20201216162406940" style="zoom:67%;" />

<img src="Week4%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E6%9D%83%E5%80%BC%E5%88%9D%E5%A7%8B%E5%8C%96.assets/image-20201216162609237.png" alt="image-20201216162609237" style="zoom:80%;" />

$\boldsymbol{D}(\boldsymbol{W})=\frac{\mathbf{1}}{\boldsymbol{n}} \Rightarrow \operatorname{std}(\mathrm{W})=\sqrt{\frac{1}{n}}$

![image-20201216163704194](Week4%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E6%9D%83%E5%80%BC%E5%88%9D%E5%A7%8B%E5%8C%96.assets/image-20201216163704194.png)

++tanh

![image-20201216163818127](Week4%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E6%9D%83%E5%80%BC%E5%88%9D%E5%A7%8B%E5%8C%96.assets/image-20201216163818127.png)

### Xavier

$n_{i} * D(W)=1$
$n_{i+1} * D(W)=1$
$\Rightarrow D(W)=\frac{2}{n_{i}+n_{i+1}}$

$\boldsymbol{W} \sim \boldsymbol{U}[-\boldsymbol{a}, \boldsymbol{a}]$
$\boldsymbol{D}(\boldsymbol{W})=\frac{(-\boldsymbol{a}-\boldsymbol{a})^{2}}{12}=\frac{(2 \boldsymbol{a})^{2}}{12}=\frac{\boldsymbol{a}^{2}}{3}$
$\frac{2}{\boldsymbol{n}_{i}+\boldsymbol{n}_{i+1}}=\frac{\boldsymbol{a}^{2}}{3} \Rightarrow \boldsymbol{a}=\frac{\sqrt{6}}{\sqrt{\boldsymbol{n}_{i}+\boldsymbol{n}_{i+1}}}$
$\Rightarrow \quad \boldsymbol{W} \sim \boldsymbol{U}\left[-\frac{\sqrt{6}}{\sqrt{n_{i}+n_{i+1}}}, \frac{\sqrt{6}}{\sqrt{n_{i}+n_{i+1}}}\right]$



具有激活函数时如何初始化

![image-20201216163313079](Week4%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E6%9D%83%E5%80%BC%E5%88%9D%E5%A7%8B%E5%8C%96.assets/image-20201216163313079.png)

#### Xavier Initialization

方差一致性:保持数据尺度维持在恰当范围,通常方差为1激活函数:饱和函数,如Sigmoid，Tanh

#### 手工计算

tanh_gain 激活函数的增益：数据输入到激活函数后，标准差的变化

![image-20201216164020150](Week4%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E6%9D%83%E5%80%BC%E5%88%9D%E5%A7%8B%E5%8C%96.assets/image-20201216164020150.png)

#### PyTorch Xavier

![image-20201216164139615](Week4%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E6%9D%83%E5%80%BC%E5%88%9D%E5%A7%8B%E5%8C%96.assets/image-20201216164139615.png)

参考文献:《Understanding the difficulty of training deep feedforward neural networks》

#### 但是如果用的ReLU

![image-20201216164324086](Week4%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E6%9D%83%E5%80%BC%E5%88%9D%E5%A7%8B%E5%8C%96.assets/image-20201216164324086.png)

### ### Kaiming 初始化

方差一致性: 保持数据尺度维持在恰当范围, 通常方差为1
激活函数:ReLU及其变种

$\mathrm{D}(W)=\frac{2}{n_{i}}$
$\mathrm{D}(W)=\frac{2}{\left(1+\mathrm{a}^{2}\right) * n_{i}}$
$\operatorname{std}(W)=\sqrt{\frac{2}{\left(1+\mathrm{a}^{2}\right) * n_{i}}}$

#### 手工计算

![image-20201216164437430](Week4%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E6%9D%83%E5%80%BC%E5%88%9D%E5%A7%8B%E5%8C%96.assets/image-20201216164437430.png)

#### PyTorch Kaiming

![image-20201216164538666](Week4%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E6%9D%83%E5%80%BC%E5%88%9D%E5%A7%8B%E5%8C%96.assets/image-20201216164538666.png)

1.Xavier均匀分布

2.Xavier正态分布

3.Kaiming均匀分布

4.Kaiming正态分布

5.均匀分布

6.正态分布

7.常数分布

8.正交矩阵初始化

9.单位矩阵初始化

10.稀疏矩阵初始化

### nn.init.calculate_gain
主要功能: 计算激活函数的方差变化尺度

主要参数
nonlinearity: 激活函数名称
param:激活函数的参数，如Leaky ReLU的negative_slop

https://pytorch.org/docs/stable/nn.init.html

![image-20201216164655710](Week4%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E6%9D%83%E5%80%BC%E5%88%9D%E5%A7%8B%E5%8C%96.assets/image-20201216164655710.png)