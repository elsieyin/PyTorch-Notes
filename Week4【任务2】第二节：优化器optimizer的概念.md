Week4【任务2】第二节：优化器optimizer的概念

pytorch的优化器:管理并更新模型中可学习参数的值，使得模型输出更接近真实标签

导数: 函数在指定坐标轴上的变化率
方向导数: 指定方向上的变化率
梯度: 一个向量,方向为方向导数取得最大值的方向

基本属性
defaults : 优化器超参数
state : 参数的缓存，如momentum的缓存
params_groups : 管理的参数组
_step_count : 记录更新次数,学习率调整中使用

```python
class Optimizer(object):
    def __init __(self, params ,defaults):
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.param_groups = []
        param_groups = [{'params': param_group}]
```

基本方法
zero_grad() : 清空所管理参数的梯度

pytorch特性 : 张量梯度不自动清零

```python
class Optimizer(object):
	def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
            	if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
```

step() : 执行一步更新

add_param_group() : 添加参数组

```python
class Optimizer(object):
	def add_param_group (self, param_group):
        for group in self.param_groups:
            param_set.update(set(group['params']))
        self.param_groups.append(param_group)
```

state_dict() :  获取优化器当前状态信息字典
load_state_dict() : 加载状态信息字典

```python
class Optimizer(object):
	def state_dict(self):
		return {
			'state': packed_state,
			'param_groups': param_groups,
        }
def load_state_dict(self, state_dict):
```



#### optimizer

![image-20201231090817753](Week4%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E4%BC%98%E5%8C%96%E5%99%A8optimizer%E7%9A%84%E6%A6%82%E5%BF%B5.assets/image-20201231090817753.png)

SGD 类是继承于 optimizer 的，所以我们将代码运行到父类初始化的这一行，点击步入，看看是如何初始化的：

![image-20201231091056299](Week4%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E4%BC%98%E5%8C%96%E5%99%A8optimizer%E7%9A%84%E6%A6%82%E5%BF%B5.assets/image-20201231091056299.png)

![image-20201231091142644](Week4%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E4%BC%98%E5%8C%96%E5%99%A8optimizer%E7%9A%84%E6%A6%82%E5%BF%B5.assets/image-20201231091142644.png)

初始化 SGD 的时候传入了一个形参：`optim.SGD(net.parameters(), lr=LR, momentum=0.9)`，这里的`net.parameters()` 就是神经网络的每层的参数，SGD 在初始化的时候， 会把这些参数以参数组的方式再存起来，上图中的 params 就是神经网络每一层的参数

跳回

![image-20201231091350231](Week4%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E4%BC%98%E5%8C%96%E5%99%A8optimizer%E7%9A%84%E6%A6%82%E5%BF%B5.assets/image-20201231091350231.png)

#### step

<img src="Week4%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E4%BC%98%E5%8C%96%E5%99%A8optimizer%E7%9A%84%E6%A6%82%E5%BF%B5.assets/image-20201230191357961.png" alt="image-20201230191357961" style="zoom:80%;" />

#### zero_grad

![image-20201230191942626](Week4%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E4%BC%98%E5%8C%96%E5%99%A8optimizer%E7%9A%84%E6%A6%82%E5%BF%B5.assets/image-20201230191942626.png)

<img src="Week4%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E4%BC%98%E5%8C%96%E5%99%A8optimizer%E7%9A%84%E6%A6%82%E5%BF%B5.assets/image-20201230192003656.png" alt="image-20201230192003656" style="zoom:67%;" />

#### add_param_group

![image-20201230192555058](Week4%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E4%BC%98%E5%8C%96%E5%99%A8optimizer%E7%9A%84%E6%A6%82%E5%BF%B5.assets/image-20201230192555058.png)

![image-20201230192624263](Week4%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E4%BC%98%E5%8C%96%E5%99%A8optimizer%E7%9A%84%E6%A6%82%E5%BF%B5.assets/image-20201230192624263.png)

#### state_dict

![image-20201230192736081](Week4%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E4%BC%98%E5%8C%96%E5%99%A8optimizer%E7%9A%84%E6%A6%82%E5%BF%B5.assets/image-20201230192736081.png)

#### load_state_dict

![image-20201230193103682](Week4%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E4%BC%98%E5%8C%96%E5%99%A8optimizer%E7%9A%84%E6%A6%82%E5%BF%B5.assets/image-20201230193103682.png)

![image-20201230193117248](Week4%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E4%BC%98%E5%8C%96%E5%99%A8optimizer%E7%9A%84%E6%A6%82%E5%BF%B5.assets/image-20201230193117248.png)