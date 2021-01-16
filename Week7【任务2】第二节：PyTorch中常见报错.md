Week7【任务2】第二节：PyTorch中常见报错

共同贡献 PyTorch 常见错误与坑汇总文档： https://shimo.im/docs/PvgHytYygPVGJ8Hv/《 PyTorch 常见报错 坑汇总》



```python
# 1.报错：
ValueError: num_samples should be a positive integer value, but got num_samples =0
```


可能的原因：传入的 Dataset 中的 len(self.data_info)==0 即传入该 dataloader 的 dataset 里没
有数据
解决方法：
1.检查 dataset 中的路径
2.检查 Dataset 的 len 函数为何输出为零

```python
# 2.报错：
TypeError : pic should be PIL Image or ndarray . Got <class torch.Tensor>
```

可能的原因：当前操作需要 PIL Image 或 ndarray 数据类型，但传入了 Tensor
解决方法：
1.检查 transform 中是否存在两次 ToTensor 方法
2.检查 transform 中每一个操作的数据类型变化

```python
# 3.报错：
RuntimeError : invalid argument 0: Sizes of tensors must match except in
dimension 0. Got 93 and 89 in dimension 1 at
/User/soumith/code/builder/pytorch-src/aten/src/TH/generic/THTensorMath.cpp:3616
```

可能的原因：dataloader 的 getitem 函数中，返回的图片形状不一致，导致无法 stack
解决方法：检查__getitem__ 函数中的操作

```python
# 4.报错：
conv:RuntimeError : Given groups=1, weight of size 6 1 5 5, expected input[16, 3,
32, 32] to have 1 channels, but got 3 channels instead
linear:RuntimeError : size mismatch, m1: [16 x 576], m2: [400 x 120]
at ../aten/src/TH/generic/THTensorMath.cpp:752
```

可能的原因：网络层输入数据与网络的参数不匹配
解决方法：
1.检查对应网络层前后定义是否有误
2.检查输入数据 shape

```python
# 5.报错：
AttributeError : DataParallel ' object has no attribute '
```

可能的原因：并行运算时，模型被 dataparallel 包装，所有 module 都增加一个属性 module.  因此
需要通过 net.module.linear 调用
解决方法：
1.网络层前加入 module.

```python
# 6.报错：
RuntimeError: Attempting to deserialize object on a CUDA device but
torch.cuda.is_available () is False. If you are running on a CPU only machine, please
use torch.load with map_location=torch.device('cpu') to map your storages to the
CPU.
```

可能的原因：
gpu 训练的模型保存后，在无 gpu 设备上无法直接加载
解决方法：
1.需要设置 map_location="cpu"

```python
# 7.报错：
AttributeError: Can't get attribute 'FooNet2' on <module '__main__' from'
```

可能的原因：保存的网络模型在当前python 脚本中没有定义
解决方法：
1.提前定义该类

```python
# 8.报错：
RuntimeError: Assertion `cur_target >= 0 && cur_target < n_classes' failed.
at ../aten/src/THNN/generic/ClassNLLCriterion.c:94
```

可能的原因：
1.标签数大于等于类别数量，即不满足 cur_target < n_classes，通常是因为标签从 1 开始而不是
从 0 开始
解决方法：
1.修改 label 从 0 开始

```python
# 9.报错：
RuntimeError: expected device cuda:0 and dtype Long but got device cpu and
dtype Long
Expected object of backend CPU but got backend CUDA for argument #2 'weight’
```

可能的原因：需计算的两个数据不在同一个设备上
解决方法：采用 to 函数将数据迁移到同一个设备

```python
# 10.报错：
RuntimeError: DataLoader worker (pid 27) is killed by signal: Killed. Details are lost
due to multiprocessing. Rerunning with num_workers=0 may give better error
```

可能原因：内存不够（不是 gpu 显存，是内存）
解决方法：申请更大内存