Week6【任务2】第一节：Batch Normalization

lesson-26

#### Batch Normalization：批标准化

批：一批数据，通常为mini-batch
标准化：0均值，1方差
#### 优点:
1.可以用更大学习率，加速模型收敛
2.可以不用精心设计权值初始化
3.可以不用dropout或较小的dropout
4.可以不用L2或者较小的weight decay
5.可以不用LRN(Iocal response normalization)

《Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift》

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9ABatch%20Normalization.assets/image-20210107143058088.png" alt="image-20210107143058088" style="zoom: 30%;" />

affine transform 增强 Capacity

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9ABatch%20Normalization.assets/image-20210107143143517.png" alt="image-20210107143143517" style="zoom: 67%;" />
$$
\begin{array}{l}
\mathrm{H}_{11}=\sum_{i=0}^{n} X_{i} * W_{1 i} \\
\mathrm{D}\left(\mathrm{H}_{11}\right)=\sum_{i=0}^{n} D\left(X_{i}\right) * D\left(W_{1 i}\right) \\
=n *(1 * 1) \\
=n \\
\operatorname{std}\left(\mathrm{H}_{11}\right)=\sqrt{\mathrm{D}\left(\mathrm{H}_{11}\right)}=\sqrt{n} \\
\mathrm{D}\left(\mathrm{H}_{1}\right)=n * D(X) * D(W)=1 \\
D(W)=\frac{1}{n} \Rightarrow \operatorname{std}(W)=\sqrt{\frac{1}{n}}
\end{array}
$$

#### bn_and_initialize

1.不进行权值初始化

![image-20210107144450656](Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9ABatch%20Normalization.assets/image-20210107144450656.png)

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9ABatch%20Normalization.assets/image-20210107144121512.png" alt="image-20210107144121512" style="zoom: 50%;" />

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9ABatch%20Normalization.assets/image-20210107144512013.png" alt="image-20210107144512013" style="zoom: 50%;" />

2.使用正态分布数据初始化 => 35层NAN

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9ABatch%20Normalization.assets/image-20210107144653421.png" alt="image-20210107144653421" style="zoom:55%;" />

3.因为relu，所以用kaiming初始化

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9ABatch%20Normalization.assets/image-20210107144837038.png" alt="image-20210107144837038" style="zoom: 60%;" />

4.用bn层，std 保持的很好

![image-20210107145103479](Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9ABatch%20Normalization.assets/image-20210107145103479.png)

5.用bn不用初始化，一样保持的很好

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9ABatch%20Normalization.assets/image-20210107145158005.png" alt="image-20210107145158005" style="zoom:67%;" />

#### bn_application.py

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9ABatch%20Normalization.assets/image-20210107161802123.png" alt="image-20210107161802123" style="zoom: 50%;" />

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9ABatch%20Normalization.assets/image-20210107161833235.png" alt="image-20210107161833235" style="zoom: 50%;" />

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9ABatch%20Normalization.assets/image-20210107161913708.png" alt="image-20210107161913708" style="zoom:70%;" />

![image-20210107161956589](Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9ABatch%20Normalization.assets/image-20210107161956589.png)

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9ABatch%20Normalization.assets/image-20210107162027011.png" alt="image-20210107162027011" style="zoom: 50%;" />

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9ABatch%20Normalization.assets/image-20210107162120629.png" alt="image-20210107162120629" style="zoom: 67%;" />

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9ABatch%20Normalization.assets/image-20210107162206157.png" alt="image-20210107162206157" style="zoom:67%;" />

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9ABatch%20Normalization.assets/image-20210107162303162.png" alt="image-20210107162303162" style="zoom: 60%;" />



有初始化的LeNet

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9ABatch%20Normalization.assets/image-20210107162638608.png" alt="image-20210107162638608" style="zoom:67%;" />

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9ABatch%20Normalization.assets/image-20210107162618249.png" alt="image-20210107162618249" style="zoom:60%;" />

BN LeNet

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9ABatch%20Normalization.assets/image-20210107162734217.png" alt="image-20210107162734217" style="zoom:67%;" />

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9ABatch%20Normalization.assets/image-20210107162757426.png" alt="image-20210107162757426" style="zoom: 60%;" />

_BatchNorm
nn.BatchNorm1d 
nn.BatchNorm2d
nn.BatchNorm3d
参数:
num_features：一个样本特征数量(最重要)
e p s：分母修正项
momentum：指数加权平均估计当前mean/var
affine：是否需要affine transform
track_running _stats：是训练状态，还是测试状态

```python
__init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
```

主要属性:
running mean:均值
running _var:方差
weight: affine transform 中的 gamma
bias: affine transform 中的 beta
$$
\begin{aligned}
\widehat{x}_{i} & \leftarrow \frac{x_{i}-\mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^{2}+\epsilon}} \\
y_{i} & \leftarrow \gamma \widehat{x}_{i}+\beta \equiv \mathrm{BN}_{\gamma, \beta}\left(x_{i}\right)
\end{aligned}
$$
训练：均值和方差采用指数加权平均计算
测试：当前统计值

running_mean = (1 momentum) * pre_running_mean + momentum * mean_t
running_var = (1 momentum) * pre_running_var + momentum * var_t

#### nn.BatchNorm1d

![image-20210107163329175](Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9ABatch%20Normalization.assets/image-20210107163329175.png)

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9ABatch%20Normalization.assets/image-20210107163415746.png" alt="image-20210107163415746" style="zoom: 40%;" />

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9ABatch%20Normalization.assets/image-20210107163427234.png" alt="image-20210107163427234" style="zoom: 60%;" />

可debug：

![image-20210107193919198](Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9ABatch%20Normalization.assets/image-20210107193919198.png)

#### BatchNorm2d

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210107194055468.png" alt="image-20210107194055468" style="zoom: 67%;" />

#### BatchNorm3d

![image-20210107194144785](Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9ABatch%20Normalization.assets/image-20210107194144785.png)