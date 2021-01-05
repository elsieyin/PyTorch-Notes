Week5【任务3】第一节：hook函数与CAM

Hook函数机制: 不改变主体，实现额外功能，像一个挂件，挂钩，hook
1. torch.Tensor.register_hook(hook)
2. torch.nn.Module.register_forward_hook
3. torch.nn.Module.register_forward_pre_hook
4. torch.nn.Module.register_backward_hook

#### **1.Tensor.register_hook**
功能：注册一个反向传播hook函数
Hook函数仅一个输入参数，为张量的梯度
`hook(grad)--> Tensor or None`
$$
\begin{array}{l}
y=(x+w) \times (w+1) \\
a=x+w \quad b=w+1 \\
y=a \times b \\
\frac{\partial y}{\partial w}=\frac{\partial y}{\partial a} \frac{\partial a}{\partial w}+\frac{\partial y}{\partial b} \frac{\partial b}{\partial w} \\
\quad=b \times 1+a \times 1 \\
\quad=b+a \\
\quad=(w+1)+(x+w) \\
\quad=2 \times w+x+1 \\
\quad=2 \times 1+2+1=5
\end{array}
$$
<img src="Week5%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Ahook%E5%87%BD%E6%95%B0%E4%B8%8ECAM.assets/image-20210104192909444.png" alt="image-20210104192909444" style="zoom:50%;" />

<img src="Week5%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Ahook%E5%87%BD%E6%95%B0%E4%B8%8ECAM.assets/image-20210104193257166.png" alt="image-20210104193257166" style="zoom:80%;" />

<img src="Week5%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Ahook%E5%87%BD%E6%95%B0%E4%B8%8ECAM.assets/image-20210104193436296.png" alt="image-20210104193436296" style="zoom:67%;" />

<img src="Week5%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Ahook%E5%87%BD%E6%95%B0%E4%B8%8ECAM.assets/image-20210104193553513.png" alt="image-20210104193553513" style="zoom: 80%;" />

<img src="Week5%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Ahook%E5%87%BD%E6%95%B0%E4%B8%8ECAM.assets/image-20210104195048401.png" alt="image-20210104195048401" style="zoom: 67%;" />

#### 2.Module.register_forward_hook
功能：注册module前向传播的hook函数
参数:
module：当前网络层
input ：当前网络层输入数据
output：当前网络层输出数据
`hook(module, input, output) --> None`

![image-20210104195524026](Week5%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Ahook%E5%87%BD%E6%95%B0%E4%B8%8ECAM.assets/image-20210104195524026.png)

#### 3.Module.register_forward_pre_hook

功能：注册module 前向传播 前 的 hook 函数
参数： module；input

`hook(module, input)--> None`
<img src="Week5%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Ahook%E5%87%BD%E6%95%B0%E4%B8%8ECAM.assets/image-20210104194022669.png" alt="image-20210104194022669"  />
<img src="Week5%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Ahook%E5%87%BD%E6%95%B0%E4%B8%8ECAM.assets/image-20210104200154709.png" alt="image-20210104200154709" style="zoom:80%;" />

#### <img src="Week5%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Ahook%E5%87%BD%E6%95%B0%E4%B8%8ECAM.assets/image-20210104200301395.png" alt="image-20210104200301395" style="zoom: 50%;" />

<img src="Week5%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Ahook%E5%87%BD%E6%95%B0%E4%B8%8ECAM.assets/image-20210104200327019.png" alt="image-20210104200327019" style="zoom: 50%;" />

#### debug

![image-20210105162319736](Week5%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Ahook%E5%87%BD%E6%95%B0%E4%B8%8ECAM.assets/image-20210105162319736.png)

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210105161929180.png" alt="image-20210105161929180" style="zoom: 50%;" />

<img src="file://C:\Users\86182\AppData\Roaming\Typora\typora-user-images\image-20210105195024571.png?lastModify=1609850779" alt="image-20210105195024571" style="zoom: 50%;" />

<img src="file://C:\Users\86182\AppData\Roaming\Typora\typora-user-images\image-20210105195302522.png?lastModify=1609850798" alt="image-20210105195302522" style="zoom:50%;" />

#### eg. Alexnet

**feature map visualization**

![image-20210105204812330](Week5%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Ahook%E5%87%BD%E6%95%B0%E4%B8%8ECAM.assets/image-20210105204812330.png)

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210105204851397.png" alt="image-20210105204851397" style="zoom:70%;" />

#### 4.Module.register_backward_hook

功能：注册module反向传播的hook函数
参数：
module：当前网络层
grad_input：当前网络层输入梯度数据
grad_output：当前网络层输出梯度数据

`hook(module, grad_input , grad_output ) --> Tensor or None`



#### CAM and Grad-CAM

![image-20210104193909422](Week5%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Ahook%E5%87%BD%E6%95%B0%E4%B8%8ECAM.assets/image-20210104193909422.png)

CAM：《Learning Deep Features for Discriminative Localization》

#### Grad -CAM: CAM改进版,利用梯度作为特征图权重

![image-20210104195546314](Week5%E3%80%90%E4%BB%BB%E5%8A%A13%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Ahook%E5%87%BD%E6%95%B0%E4%B8%8ECAM.assets/image-20210104195546314.png)

Grad-CAM：《 Grad CAM: Visual Explanations from Deep Networks via Gradient-based Localization 》

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210105220327601.png" alt="image-20210105220327601" style="zoom: 67%;" />

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210105220332247.png" alt="image-20210105220332247" style="zoom:67%;" />

分析与代码：
https://zhuanlan.zhihu.com/p/75894080