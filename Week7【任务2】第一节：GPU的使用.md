Week7【任务2】第一节：GPU的使用

CPU (Central Processing Unit，中央处理器):主要包括控制器和运算器
GPU(Graphics Processing Unit,图形处理器):处理统一的，无依赖的大规模数据运算

![image-20210113101512320](Week7%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9AGPU%E7%9A%84%E4%BD%BF%E7%94%A8.assets/image-20210113101512320.png)

![image-20210113101548853](Week7%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9AGPU%E7%9A%84%E4%BD%BF%E7%94%A8.assets/image-20210113101548853.png)

data:
1.Tensor
2.Module

to函数:转换数据类型/设备

1.tensor.to(\*args, \*\*kwargs)
2.module.to(\*args, \*\*kwargs)
区别：张量不执行inplace，模型执行inplace

```python
x = torch.ones((3,3))
x = x.to(torch.float64)

x = torch.ones((3,3))
x = x.to("cuda")

linear = nn.Linear(2,2)
linear.to(torch.double)
gpu1 = torch.device("cuda")
linear.to(gpu1)
```

torch.cuda常用方法

1.torch.cuda.device_count(): 计算当前可见可用gpu数
2.torch.cuda.get_device_name(): 获取gpu名称
3.torch.cuda.manual_seed(): 为当前gpu设置随机种子
4.torch.cuda.manual_seed_all(): 为所有可见可用gpu设置随机种子
5.torch.cuda.set_device(): 设置主gpu为哪一个物理gpu(不推荐)
推荐: os.environ.setdefault("CUDA_VISIBLE_DEVICES"，"2,3")

![image-20210113102042656](Week7%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9AGPU%E7%9A%84%E4%BD%BF%E7%94%A8.assets/image-20210113102042656.png)

![image-20210113102144869](Week7%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9AGPU%E7%9A%84%E4%BD%BF%E7%94%A8.assets/image-20210113102144869.png)

多gpu运算的分发并行机制
torch.nn.DataParallel
功能: 包装模型，实现分发并行机制
主要参数:
module: 需要包装分发的模型
device_ids: 可分发的gpu，默认分发到所有可见可用gpu
output_device: 结果输出设备

```python
torch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)
```

#### 多gpu运算的分发并行机制
batch size in forward:  8
batch size in forward:  8
model outputs.size: torch.Size([16，3])
CUDA_VISIBLE_DEVICES : 2,3
device_count : 2
gpu free memory:  [10362, 10058, 9990, 9990]
batch size in forward:  4
batch size in forward:  4
batch size in forward:  4
batch size in forward:  4
model outputs.size:  torch.Size([16，3])
CUDA_VISIBLE_DEVICES : 0,1,3,2
device_count : 4

#### 查询当前 gpu 内存剩余

```python
def get_gpu_memory():
	import os
	os.system('nvidia -smi -q -d Memory | grep -A4 GPU | grep Free > tmp.txt')
	memory_gpu = [int(x.split()[2]) for x in open('tmp.txt', 'r').readlines()]
	os.system('rm tmp.txt')
	return memory_gpu
```

```python
example:
gpu_memory = get_gpu_memory()
gpu_list = np.argsort(gpu_memory)[::-1]
gpu_list_str = ','.join(map(str, gpu_list))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)

print("\ngpu free memory: {}".format(gpu_memory))
print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

>>> gpu free memory: [10362, 10058, 9990, 9990]
>>> CUDA_VISIBLE_DEVICES :0,1,3,2
```

#### gpu模型加载

```python
# 报错1:
RuntimeError: Attempting to deserialize object on a CUDA device but
    torch.cuda.is_available() is False. If you are running on a CPU only machine,
    please use torch.load with map_location=torch.device('cpu') to map your storages
    to the CPU.
    
# 解决
torch.load(path_state_dict, map_location="cpu")
```

```python
# 报错2 
RuntimeError : Error(s) in loading state_dict for FooNet Missing key(s) in
state_dict: "linears.0.weight", "linears.1.weight", "linears.2.weight".
Unexpected key(s) in state_dict : "module.linears.0.weight",
"module.linears.1.weight", "module.linears.2.weight".

# 解决
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict_load.items():
	namekey = k[7:] if k.startswith ('module.') else k
	new_state_dict[namekey] = v
```

