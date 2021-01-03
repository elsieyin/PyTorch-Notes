#### **「5 nn.L1Loss」**

这个用于回归问题，用来计算inputs与target之差的绝对值

``````python
nn.L1Loss( size_average =None, reduce=None, reduction='mean’)
``````

$l_{n}=\left|x_{n}-y_{n}\right|$

上面的 size_average 和 reduce 不用再关注，即将淘汰。而 reduction 这个三种模式，其实和上面的一样。

![image-20201226111229074](Week4%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Apytorch%E7%9A%8414%E7%A7%8D%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.assets/image-20201226111229074.png)

![image-20201225200748156](Week4%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Apytorch%E7%9A%8414%E7%A7%8D%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.assets/image-20201225200748156.png)

#### **「6 nn.MSE」**

这个也是用于回归问题，计算inputs与target之差的平方

```python
nn.MSELoss(size_average =None, reduce=None, reduction='mean’)
```

$l_{n}=\left(x_{n}-y_{n}\right)^{2}$

reduction:计算模式，可为none/sum/mean
none-逐个元素计算
sum-所有元素求和，返回标量
mean-加权平均，返回标量

#### **「7 nn.SmoothL1Loss」**

这是平滑的L1Loss（回归问题）

```python
nn.SmoothL1Loss(size_average =None, reduce=None, reduction='mean’)
```

$\operatorname{loss}(x, y)=\frac{1}{n} \sum_{i} z_{i}$
$z_{i}=\left\{\begin{array}{ll}
0.5\left(x_{i}-y_{i}\right)^{2}, & \text { if }\left|x_{i}-y_{i}\right|<1 \\
\left|x_{i}-y_{i}\right|-0.5, & \text { otherwise }
\end{array}\right.$

![图片](https://mmbiz.qpic.cn/mmbiz_png/210IyDic7racmzhFcYJ3xGkoS1LtVdaKKXj9kiaJRbFH2zm67GQScACbmpiakHfE0Rdxkj8EE7epZVvCnicwQ27SvQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

采用这种平滑的损失函数可以减轻离群点带来的影响。

#### **「8 nn.PoissonNLLLoss」**

功能：泊松分布的负对数似然损失函数，分类里面如果发现数据的类别服从泊松分布，可以使用这个损失函数

```python
nn.PoissonNLLLoss(log_input =True, full=False, size_average =None, eps=1e-08, reduce=None, reduction='mean')
```

log_input = True
loss(input, target) = exp (input) target * input
log_input = False
loss(input, target) = input - target * log(input + eps)

- log_intput: 输入是否为对数形式，决定我们的计算公式。若为True， . 若为False，

- full: 计算所有loss，默认为False，这个一般不用管

- eps: 修正项，避免log(input) 为nan

  

  ![image-20201225202250392](Week4%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Apytorch%E7%9A%8414%E7%A7%8D%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.assets/image-20201225202250392.png)

#### **「9 nn.KLDivLoss」**

功能：计算 KLD， KL 散度，相对熵，注意：需要提前将输入计算 log-probabilities，如通过 nn.logsoftmax()

![图片](https://mmbiz.qpic.cn/mmbiz_png/210IyDic7racmzhFcYJ3xGkoS1LtVdaKKGqdtdV9qxx5S5akEf2x5e6xLZaRNOt3yhNZLheickRF8hicDFlmx3FNg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

上面的 Pytorch 里面的计算和我们原来公式里面的计算还有点不太一样，所以得自己先logsoftmax()，完成转换为分布然后转成对数才可以。这里的 reduction 还多了一种计算模式叫做 batchmean，是按照 batchsize 的大小求平均值。

![image-20201225204316579](Week4%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Apytorch%E7%9A%8414%E7%A7%8D%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.assets/image-20201225204316579.png)

#### **「10 nn.MarginRankingLoss」**

功能：计算两个向量之间的相似度，用于排序任务。特别说明，该方法计算两组数据之间的差异，也就是每个元素两两之间都会计算差异，返回一个 n*n 的 loss 矩阵。类似于相关性矩阵那种。

```python
nn.MarginRankingLoss(margin=0.0, size_average =None, reduce=None, reduction='mean')
```

margin 表示边界值，x1 与 x2 之间的差异值。这里的计算公式如下：

$\operatorname{loss}(x, y)=\max (0,-y * {(x 1-x 2)}+\operatorname{margin})$

- y=1时， 希望x1比x2大， 当x1>x2时，不产生loss
- y=-1时， 希望x2比x1大， 当x2>x1时， 不产生loss

![image-20201225204949909](Week4%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Apytorch%E7%9A%8414%E7%A7%8D%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.assets/image-20201225204949909.png)

**「11 nn.MultiLabelMarginLoss」**

功能：多标签边界损失函数， 这是一个多标签分类，就是一个样本可能属于多个类，和多分类任务还不一样。（多标签问题）

举例:四分类任务，样本x属于0类和3类，标签:[0,3.-1,01]，不是[1,0,0,1]
主要参数: 
reduction :计算模式，可为none/sum/mean

```python
nn.MultiLabelMarginLoss(size_average=None, reduce=None,reduction='mean')
```

$$\operatorname{loss}(x, y)=\sum_{i j} \frac{\max (0,1-(x[y[j]]-x[i]))}{x \cdot \operatorname{size}(0)}$$

这里的 i 取值从 0 到输出的维度减 1，j 取值也是 0 到 y 的维度减 1，对于所有的 i 和 j，i 不等于 y[j]，也就是标签所在的神经元去减掉那些非标签所在的神经元。

![image-20201225205933111](Week4%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Apytorch%E7%9A%8414%E7%A7%8D%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.assets/image-20201225205933111.png)

#### **「12 nn.SoftMarginLoss」**

```python
nn.SoftMarginLoss(size_average=None, reduce=None, reduction='mean')
```

$\operatorname{loss}(x, y)=\sum_{i} \frac{\log (1+\exp (-y[i] * x[i]))}{\text { x.nelement }()}$

功能：计算二分类的 logistic 损失
主要参数：
•reduction ：计算模式，可为 none/sum/mean

![image-20201225210528623](Week4%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Apytorch%E7%9A%8414%E7%A7%8D%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.assets/image-20201225210528623.png)

#### **「13 nn.MultiLabelSoftMarginLoss」**

```python
nn.MultiLabelSoftMarginLoss(weight=None, size_average =None, reduce=None,reduction='mean')
```

功能：SoftMarginLoss 多标签版本 （多标签问题）

$\operatorname{loss}(x, y)=-\frac{1}{C} * \sum_{i} y[i] * \log \left((1+\exp (-x[i]))^{-1}\right)+({1-y[i]}){} * \log \left(\frac{\exp (-x[i])}{(1+\exp (-x[i]))}\right)$

![image-20201230112021962](Week4%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Apytorch%E7%9A%8414%E7%A7%8D%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.assets/image-20201230112021962.png)

<img src="Week4%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Apytorch%E7%9A%8414%E7%A7%8D%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.assets/image-20201230112045183.png" alt="image-20201230112045183" style="zoom: 67%;" />

#### **「14 nn.MultiMarginLoss（hingLoss）」**

功能：计算多分类的折页损失（多分类问题）

```python
nn.MultiMarginLoss(p=1, margin=1.0, weight=None, size_average =None, reduce=None, reduction='mean')
```

参数：
p: 可选1或2
weight: 各类别的loss设置权值
margin :边界值
reduction :计算模式，可为none/sum/mean
$$
\operatorname{loss}(x, y)=\frac{\left.\sum_{i} \max (0, \operatorname{margin}-x[y]+x[i])\right)^{p}}{x \cdot \operatorname{size}(0)}
$$
![image-20201230112715497](Week4%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Apytorch%E7%9A%8414%E7%A7%8D%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.assets/image-20201230112715497.png)

这里的 x, y 是 0 - 神经元个数减 1，并且对于所以 i 和 j，i 不等于 y[j]。这里就类似于 hing loss 了，这里的 x[y] 表示标签所在的神经元，x[i] 表示非标签所在的神经元。

![image-20201230112855722](Week4%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Apytorch%E7%9A%8414%E7%A7%8D%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.assets/image-20201230112855722.png)

#### **「15 nn.TripletMarginLoss」**

功能：计算三元组损失，人脸验证中常用

```python
nn.TripletMarginLoss(margin=1.0, p=2.0, eps=1e-06, swap=False, size_average =None, reduce=None, reduction='mean')
```

主要参数:
p:范数的阶，默认为2
margin :边界值
reduction :计算模式，可为none/sum/mean

![image-20201230113430801](Week4%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Apytorch%E7%9A%8414%E7%A7%8D%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.assets/image-20201230113430801.png)
$$
L(a, p, n)=\max \left\{d\left(a_{i}, p_{i}\right)-d\left(a_{i}, n_{i}\right)+\operatorname{margin}, 0\right\}
\\
d\left(x_{i}, y_{i}\right)=\left\|\mathbf{x}_{i}-\mathbf{y}_{i}\right\|_{p}
$$
做人脸识别训练模型的时候，往往需要把训练集做成三元组 (A, P, N)， A 和 P 是同一个人，A 和 N 不是同一个，然后训练我们的模型

![图片](Week4%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Apytorch%E7%9A%8414%E7%A7%8D%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.assets/640)

我们想让模型把 A 和 P 看成一样的，也就是争取让 A 和 P 之间的距离小，而 A 和 N 之间的距离大，那么我们的模型就能够进行人脸识别任务了。

![image-20201230113301449](Week4%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Apytorch%E7%9A%8414%E7%A7%8D%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.assets/image-20201230113301449.png)

#### **「16 nn.HingeEmbeddingLoss」**

功能:计算两个输入的相似性,常用于非线性embedding和半监督学习
特别注意:输入x应为两个输入之差的绝对值
主要参数:
margin :边界值
reduction :计算模式,可为none/sum/mean

```python
nn.HingeEmbeddingLoss(margin=1.0, size_average =None, reduce=None, reduction='mean’)
```

<img src="Week4%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Apytorch%E7%9A%8414%E7%A7%8D%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.assets/image-20201230113631473.png" alt="image-20201230113631473" style="zoom: 50%;" />

![image-20201230113357229](Week4%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Apytorch%E7%9A%8414%E7%A7%8D%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.assets/image-20201230113357229.png)

#### **「17 nn.CosineEmbeddingLoss」**

功能:采用余弦相似度计算两个输入的相似性
主要参数:
margin :可取值[-1,1]，推荐为[0,0.5]
reduction :计算模式，可为none/sum/mean

```python
nn.CosineEmbeddingLoss(margin=0.0, size_average =None, reduce=None, reduction='mean')
```

$$
\begin{array}{l}
\operatorname{loss}(x, y)=\left\{\begin{array}{ll}
1-\cos \left(x_{1}, x_{2}\right), & \text { if } y=1 \\
\max \left(0, \cos \left(x_{1}, x_{2}\right)-\operatorname{margin}\right), & \text { if } y=-1
\end{array}\right. \\
\cos (\theta)=\frac{A \cdot B}{\|A \mid\| B \|}=\frac{\sum_{i=1}^{n} A_{i} \times B_{i}}{\sqrt{\sum_{i=1}^{n}\left(A_{i}\right)^{2}} \times \sqrt{\sum_{i=1}^{n}\left(B_{i}\right)^{2}}}
\end{array}
$$

![image-20201230113901195](Week4%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Apytorch%E7%9A%8414%E7%A7%8D%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.assets/image-20201230113901195.png)

#### **「18 nn.CTCLoss」**

功能:计算CTC损失,解决时序类数据的分类Connectionist Temporal Classification
主要参数:
blank : blank label
zero_infinity :无穷大的值或梯度置0
reduction :计算模式,可为none/sum/mean

```python
torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity =False)
```

![image-20201230114114811](Week4%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9Apytorch%E7%9A%8414%E7%A7%8D%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.assets/image-20201230114114811.png)