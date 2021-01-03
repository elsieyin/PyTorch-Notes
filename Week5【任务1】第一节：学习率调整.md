Week5【任务1】第一节：学习率调整

class _LRScheduler
主要方法:
optimizer: 关联的优化器
last_epoch: 记录epoch数
base_lrs: 记录初始学习率

```python
class _LRScheduler(object):
	def __init __(self, optimizer, last_epoch=-1):
```

step(): 更新下一个epoch的学习率
get_Ir(): 虚函数,计算下一个epoch的学习率

```python
class _LRScheduler(object):
	def __init __(self, optimizer, last_epoch=-1):
	def get_lr(self):
        raise NotImplementedError
```

![image-20201231145602518](Week5%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E6%95%B4.assets/image-20201231145602518.png)

![image-20201231145653053](Week5%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E6%95%B4.assets/image-20201231145653053.png)

![image-20201231150005778](Week5%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E6%95%B4.assets/image-20201231150005778.png)

![image-20201231150158296](Week5%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E6%95%B4.assets/image-20201231150158296.png)

121行打断点，step into，注意scheduler.step()位置，是在epoch的for循环中

![image-20201231151031416](Week5%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E6%95%B4.assets/image-20201231151031416.png)

param_group参数组是一个list，list中每个元素是一个字典，

### lr_decay_scheduler

#### 1.StepLR

功能: 等间隔调整学习率
主要参数:
step_size: 调整间隔数
gamma: 调整系数
调整方式: Ir = lr * gamma

```python
lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
```

![image-20201231154959345](Week5%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E6%95%B4.assets/image-20201231154959345.png)

<img src="Week5%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E6%95%B4.assets/image-20201231155018056.png" alt="image-20201231155018056" style="zoom: 67%;" />

#### 2.MultiStepLR
功能:按给定间隔调整学习率
主要参数:
milestones : 设定调整时刻数
gamma: 调整系数
调整方式:  lr = Ir * gamma

```python
lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
```

![image-20201231155335692](Week5%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E6%95%B4.assets/image-20201231155335692.png)

<img src="Week5%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E6%95%B4.assets/image-20201231155416417.png" alt="image-20201231155416417" style="zoom: 67%;" />

#### 3.ExponentialLR
功能: 按指数衰减调整学习率
主要参数:
gamma: 指数的底
调整方式： lr = lr * gamma ** epoch

```python
lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
```

![image-20201231155802315](Week5%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E6%95%B4.assets/image-20201231155802315.png)

<img src="Week5%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E6%95%B4.assets/image-20201231155815941.png" alt="image-20201231155815941" style="zoom:67%;" />

#### 4.CosineAnnealingLR
功能: 余弦周期调整学习率
主要参数:
T_max: 下降周期
eta min: 学习率下限
$$
\eta_{t}=\eta_{\min }+\frac{1}{2}\left(\eta_{\max }-\eta_{\min }\right)\left(1+\cos \left(\frac{T_{c u r}}{T_{\max }} \pi\right)\right)
$$

```python
lr_scheduler.CosineAnnealingLR(optimizer, T_max , eta_min=0, last_epoch=-1)
```

![image-20201231160338144](Week5%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E6%95%B4.assets/image-20201231160338144.png)

<img src="Week5%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E6%95%B4.assets/image-20201231160344426.png" alt="image-20201231160344426" style="zoom:67%;" />

#### 5.ReduceLRonPlateau

功能:监控指标,当指标不再变化则调整
主要参数:
mode: min/max两种模式
factor: 调整系数
patience: “耐心”，接受几次不变化
cooldown: “冷却时间”，停止监控一段时间
verbose:是否打印日志
min_lr: 学习率下限 
eps:学习率衰减最小值

```python
lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
verbose=False, threshold=0.0001, threshold_mode\'rel', cooldown=0, min_lr =0,
eps=1e-08)
```

![image-20201231161022608](Week5%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E6%95%B4.assets/image-20201231161022608.png)

#### 6.LambdaLR
功能:自定义调整策略主要参数:
lr_lambda:  function or list

```python
lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
```

![image-20201231161455125](Week5%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E6%95%B4.assets/image-20201231161455125.png)

![image-20201231162951849](Week5%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E6%95%B4.assets/image-20201231162951849.png)
<img src="Week5%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E6%95%B4.assets/image-20201231163046948.png" alt="image-20201231163046948" style="zoom: 67%;" />

学习率调整小结
1.有序调整:Step、MultiStep、Exponential 和CosineAnnealing
2.自适应调整: ReduceLROnPleateau
3.自定义调整: Lambda

学习率初始化:
1.设置较小数: 0.01、0.001、0.0001
2.搜索最大学习率:《cyclical Learning Rates for Training Neural Networks》
<img src="Week5%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E6%95%B4.assets/image-20201231162753206.png" alt="image-20201231162753206" style="zoom: 50%;" />

<img src="Week5%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E6%95%B4.assets/image-20201231162801117.png" alt="image-20201231162801117" style="zoom:67%;" />