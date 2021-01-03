#### 损失函数

损失函数：衡量模型输出与真实标签的差异。

<img src="Week4%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%EF%BC%88%E4%B8%80.assets/image-20201217105304596.png" alt="image-20201217105304596" style="zoom: 67%;" />

Loss Function: 计算一个样本的一个差异。 $Loss=f(\hat{y}, y)$
Cost Function: 计算整个训练集Loss的一个平均值。 $\cos t=\frac{1}{N} \sum_{i}^{N} f\left(\hat{y}_{i}, y_{i}\right)$
Objective Function: 这是一个更广泛的概念，在机器学习模型训练中，这是最终
一个目标，过拟合和欠拟合之间进行一个权衡。 $Obj=Cost + Regularization$

```python
class _Loss(Module):
	def init ____(self, size_average=None , reduce=None,
				reduction='mean'):
		super(_Loss, self).__init__()
		if size_average is not None or reduce is not None:
			self.reduction = _Reduction.legacy_get_string(
                size_average, reduce)
 		else:
			self.reduction = reduction
```



交叉熵 = 信息熵 + 相对熵

交叉嫡： $\mathrm{H}(P, Q)=-\sum_{i=1}^{N} P\left(x_{i}\right) \log Q\left(x_{i}\right)$
自信息 $: \mathrm{I}(x)=-\log [p(x)]$
嫡： $\mathrm{H}(\mathrm{P})=E_{x \sim p}[I(x)]=-\sum_{i}^{N} P\left(x_{i}\right) \log P\left(x_{i}\right)$
相对嫡： 
$$
\begin{array}{l}
D_{K L}(P, Q)=E_{x \sim p}\left[\log \frac{P(x)}{Q(x)}\right]
=E_{x \sim p}[\log P(x)-\log Q(x)] \\
=\sum_{i=1}^{N} P\left(x_{i}\right)\left[\log P\left(x_{i}\right)-\log Q\left(x_{i}\right)\right] \\
=\sum_{i=1}^{N} P\left(x_{i}\right) \log P\left(x_{i}\right)-\sum_{i=1}^{N} P\left(x_{i}\right) \log Q\left(x_{i}\right)\\
=\boldsymbol{H}(\boldsymbol{P}, \boldsymbol{Q})-\boldsymbol{H}(\mathrm{P})
\end{array}
$$
nn.CrossEntropyLoss: 

nn.LogSortmax() 与 nn.NLLLoss() 结合，进行交叉熵计算

- weight：各类别的 loss 设置权值

- ignore_index：忽略某个类别

- reduction：计算模式，可为 none/sum/mean，

  none - 逐个元素计算，这样有多少个样本就会返回多少个 loss。

  sum - 所有元素的 loss 求和，返回标量，

  mean - 所有元素的 loss 求加权平均，返回标量。

<img src="Week4%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%EF%BC%88%E4%B8%80.assets/image-20201217115925666.png" alt="image-20201217115925666" style="zoom: 80%;" />

<img src="Week4%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%EF%BC%88%E4%B8%80.assets/image-20201217153035258.png" alt="image-20201217153035258" style="zoom:80%;" />

<img src="Week4%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%EF%BC%88%E4%B8%80.assets/image-20201217153122938.png" alt="image-20201217153122938" style="zoom:80%;" />

#### weight

<img src="Week4%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%EF%BC%88%E4%B8%80.assets/image-20201217153407123.png" alt="image-20201217153407123" style="zoom:80%;" />

![image-20201224190209972](Week4%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%EF%BC%88%E4%B8%80.assets/image-20201224190209972.png)

![image-20201224190333605](Week4%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%EF%BC%88%E4%B8%80.assets/image-20201224190333605.png)

<img src="Week4%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%EF%BC%88%E4%B8%80.assets/image-20201224190409635.png" alt="image-20201224190409635" style="zoom: 50%;" />

<img src="Week4%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%EF%BC%88%E4%B8%80.assets/image-20201224190444221.png" alt="image-20201224190444221" style="zoom: 67%;" />

### NLLLoss

`nn.NLLoss`: 实现负对数似然函数里面的负号功能

<img src="Week4%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%EF%BC%88%E4%B8%80.assets/image-20201224191512236.png" alt="image-20201224191512236" style="zoom:67%;" />

$\ell(x, y)=L=\left\{l_{1}, \ldots, l_{N}\right\}^{1}, \quad l_{n}=-w_{y_{n}} x_{n, y_{n}}$

功能:实现负对数似然函数中的负号功能
主要参数:
weight:各类别的loss设置权值
ignore_index:忽略某个类别
reduction :计算模式，可为none/sum/mean
none-逐个元素计算
sum-所有元素求和，返回标量
mean-加权平均，返回标量

![image-20201224191027982](Week4%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%EF%BC%88%E4%B8%80.assets/image-20201224191027982.png)

### BCELoss

功能:二分类交叉嫡
注意事项:输入值取值在[0,1]
主要参数:
weight:各类别的loss设置权值
ignore_index:忽略某个类别
reduction :计算模式，可为none/ sum/mean
	none-逐个元素计算
	sum-所有元素求和，返回标量
	mean-加权平均，返回标量

<img src="Week4%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%EF%BC%88%E4%B8%80.assets/image-20201224191350883.png" alt="image-20201224191350883" style="zoom:67%;" />

$l_{n}=-w_{n}\left[y_{n} \cdot \log x_{n}+\left(1-y_{n}\right) \cdot \log \left(1-x_{n}\right)\right]$

![image-20201224191806564](Week4%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%EF%BC%88%E4%B8%80.assets/image-20201224191806564.png)

![image-20201224192237357](Week4%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%EF%BC%88%E4%B8%80.assets/image-20201224192237357.png)

<img src="Week4%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%EF%BC%88%E4%B8%80.assets/image-20201224192408701.png" alt="image-20201224192408701" style="zoom: 67%;" />

### BCEWithLogitsLoss

<img src="Week4%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%EF%BC%88%E4%B8%80.assets/image-20201224192615953.png" alt="image-20201224192615953" style="zoom:67%;" />

$l_{n}=-w_{n}\left[y_{n} \cdot \log \sigma\left(x_{n}\right)+\left(1-y_{n}\right) \cdot \log \left(1-\sigma\left(x_{n}\right)\right)\right]$

功能:结合Sigmoid与二分类交叉嫡
注意事项:网络最后不加sigmoid函数主要参数:
pos_weight : 正样本的权值
weight:各类别的loss设置权值
ignore_index:忽略某个类别
reduction :计算模式，可为none/ sum/mean
	none-逐个元素计算
	sum-所有元素求和，返回标量
	mean-加权平均，返回标量

![image-20201224193002084](Week4%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%EF%BC%88%E4%B8%80.assets/image-20201224193002084.png)

#### pos weight

![image-20201224193137600](Week4%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%EF%BC%88%E4%B8%80.assets/image-20201224193137600.png)