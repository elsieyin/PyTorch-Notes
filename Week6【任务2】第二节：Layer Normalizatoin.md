Week6【任务2】第二节：Layer Normalizatoin

lesson-27

常见的Normalization

1. Batch Normalization (BN)
2. Layer Normalization ( LN)
3. Instance Normalization (IN)
4. Group Normalization (GN)

同：
<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9ALayer%20Normalizatoin.assets/image-20210107203827649.png" alt="image-20210107203827649" style="zoom: 43%;" />

异：
均值和方差求取方式

1.Layer Normalization
起因：BN不适用于变长的网络，如RNN
思路：逐层计算均值和方差

注意事项:
1.不再有running_mean和running var
2.gamma和beta为逐元素的

《 Layer Normalization 》

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9ALayer%20Normalizatoin.assets/image-20210107204000112.png" alt="image-20210107204000112" style="zoom:50%;" />

nn.LayerNorm
主要参数:
normalized shape：该层特征形状
eps：分母修正项
elementwise_affine：是否需要affinetransform

```python
nn.LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True)
```

![image-20210107204611956](Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9ALayer%20Normalizatoin.assets/image-20210107204611956.png)

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9ALayer%20Normalizatoin.assets/image-20210107204631554.png" alt="image-20210107204631554" style="zoom: 50%;" />

![image-20210107204742304](Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9ALayer%20Normalizatoin.assets/image-20210107204742304.png)

![image-20210107204750145](Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9ALayer%20Normalizatoin.assets/image-20210107204750145.png)

![image-20210107205004290](Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9ALayer%20Normalizatoin.assets/image-20210107205004290.png)

```python
File "F:\Anaconda\envs\kaggle\lib\site-packages\torch\nn\functional.py", line 2049, in layer_norm
    torch.backends.cudnn.enabled)
RuntimeError: Given normalized_shape=[6, 3], expected input with shape [*, 6, 3], but got input of size[8, 6, 3, 4]
```

#### 2.Instance Normalization

起因:BN在图像生成（Image Generation）中不适用
思路:逐lnstance (channel) 计算均值和方差

![image-20210107210239764](Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9ALayer%20Normalizatoin.assets/image-20210107210239764.png)

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9ALayer%20Normalizatoin.assets/image-20210107210249345.png" alt="image-20210107210249345" style="zoom: 50%;" />

《Instance Normalization: The Missing Ingredient for Fast Stylization 》
《Image Style Transfer Using Convolutional Neural Networks》

nn.lnstanceNorm
主要参数:
num_features：一个样本特征数量(最重要)
eps：分母修正项
momentum：指数加权平均估计当前mean/var
affine：是否需要affine transform
track_running_stats：是训练状态，还是测试状态

```python
nn.InstanceNorm2d(num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats = False)
```

![image-20210107211026319](Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9ALayer%20Normalizatoin.assets/image-20210107211026319.png)

#### 3.Group Normalization

起因:小batch样本中，BN估计的值不准
思路:数据不够，通道来凑
注意事项:
1.不再有running_mean 和 running_var

2.gamma 和 beta 为逐通道（channel)的
应用场景：大模型(小batch size）任务
《Group Normalization 》

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9ALayer%20Normalizatoin.assets/image-20210107210603195.png" alt="image-20210107210603195" style="zoom:50%;" />

nn.GroupNorm
主要参数:
num _groups：分组数
num_channels：通道数（特征数)
eps：分母修正项
affine：是否需要 affine transform

```python
nn.GroupNorm(num_groups, num_channels, eps=1e-05, affine=True)
```

小结： BN 、 LN 、 IN 和 GN 都是为了克服 Internal Covariate Shift (ICS)

![image-20210107210810763](Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9ALayer%20Normalizatoin.assets/image-20210107210810763.png)

![image-20210107211107195](Week6%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9ALayer%20Normalizatoin.assets/image-20210107211107195.png)