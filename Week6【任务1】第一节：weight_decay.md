Regularization：减小方差的策略
误差可分解为：偏差，方差与噪声之和。即误差=偏差＋方差＋噪声之和偏差度量了学习算法的期望预测与真实结果的偏离程度，即刻画了学习算法本身的拟合能力
方差度量了同样大小的训练集的变动所导致的学习性能的变化，即刻画了数据扰动所造成的影响
噪声则表达了在当前任务上任何学习算法所能达到的期望泛化误差的下界

![image-20210106150758555](C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210106150758555.png)

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210106150814082.png" alt="image-20210106150814082" style="zoom:33%;" />

Loss Function: 计算一个样本的一个差异。 $Loss=f(\hat{y}, y)$
Cost Function: 计算整个训练集Loss的一个平均值。 $\cos t=\frac{1}{N} \sum_{i}^{N} f\left(\hat{y}_{i}, y_{i}\right)$
Objective Function: 这是一个更广泛的概念，在机器学习模型训练中，这是最终
一个目标，过拟合和欠拟合之间进行一个权衡。 $Obj=Cost + Regularization$

![image-20210106150920682](C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210106150920682.png)

目标函数(Objective Function)：$Obj=Cost + Regularization$
$$
L1  Regularization  Term: \sum_{i}^{N} \left|w_{i}\right|
\\
L2  Regularization  Term: \sum_{i}^{N} w_{i}^{2}
$$
<img src="Regularization%EF%BC%9A%E5%87%8F%E5%B0%8F%E6%96%B9%E5%B7%AE%E7%9A%84%E7%AD%96%E7%95%A5.assets/image-20210106151050549.png" alt="image-20210106151050549" style="zoom: 50%;" />

L2 Regularization = weight decay (权值衰减)
目标函数（Objective Function) ：
$$
\begin{array}{l}
\text { Obj }=\text { Cost }+\text { Regularization Term } \\
\qquad \begin{array}{l}
\text { Obj }=\text { Loss }+\frac{\lambda}{2} * \sum_{i}^{N} w_{i}^{2} \\
w_{i+1}=w_{i}-\frac{\partial O b j}{\partial w_{i}}=w_{i}-\frac{\partial L o s s}{\partial w_{i}} \\
w_{i+1}=w_{i}-\frac{\partial O b j}{\partial w_{i}}=w_{i}-\left(\frac{\partial \text { Loss }}{\partial w_{i}}+\lambda * w_{i}\right) \\
=w_{i}(1-\lambda)-\frac{ {\partial L o s s}}{\partial w_{i}}
\end{array}
\end{array}
$$
<img src="Regularization%EF%BC%9A%E5%87%8F%E5%B0%8F%E6%96%B9%E5%B7%AE%E7%9A%84%E7%AD%96%E7%95%A5.assets/image-20210106151102539.png" alt="image-20210106151102539" style="zoom:50%;" />

<img src="Regularization%EF%BC%9A%E5%87%8F%E5%B0%8F%E6%96%B9%E5%B7%AE%E7%9A%84%E7%AD%96%E7%95%A5.assets/image-20210106154736536.png" alt="image-20210106154736536" style="zoom:67%;" />

![image-20210106154758232](Regularization%EF%BC%9A%E5%87%8F%E5%B0%8F%E6%96%B9%E5%B7%AE%E7%9A%84%E7%AD%96%E7%95%A5.assets/image-20210106154758232.png)

![image-20210106154834112](Regularization%EF%BC%9A%E5%87%8F%E5%B0%8F%E6%96%B9%E5%B7%AE%E7%9A%84%E7%AD%96%E7%95%A5.assets/image-20210106154834112.png)

<img src="Regularization%EF%BC%9A%E5%87%8F%E5%B0%8F%E6%96%B9%E5%B7%AE%E7%9A%84%E7%AD%96%E7%95%A5.assets/image-20210106160509035.png" alt="image-20210106160509035" style="zoom: 50%;" />

<img src="Regularization%EF%BC%9A%E5%87%8F%E5%B0%8F%E6%96%B9%E5%B7%AE%E7%9A%84%E7%AD%96%E7%95%A5.assets/image-20210106160612987.png" alt="image-20210106160612987" style="zoom:50%;" />

![image-20210106161550195](Regularization%EF%BC%9A%E5%87%8F%E5%B0%8F%E6%96%B9%E5%B7%AE%E7%9A%84%E7%AD%96%E7%95%A5.assets/image-20210106161550195.png)