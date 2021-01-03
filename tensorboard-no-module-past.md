Week5【任务1】第二节：TensorBoard简介与安装

![图片](tensorboard-no-module-past.assets/640)

![图片](https://mmbiz.qpic.cn/mmbiz_png/210IyDic7racmzhFcYJ3xGkoS1LtVdaKKndOyFPb1kwJFBUcuzNqtr7Rxto5t5P2YBczeuaeAB6qqnLwAc7c74g/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

```python
import numpy as np
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter(comment='test_tensorboard')

for x in range(100):

    writer.add_scalar('y=2x', x * 2, x)
    writer.add_scalar('y=pow(2, x)',  2 ** x, x)
    
    writer.add_scalars('data/scalar_group', {"xsinx": x * np.sin(x),
                                             "xcosx": x * np.cos(x),
                                             "arctanx": np.arctan(x)}, x)
writer.close()
```

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210103153033519.png" alt="image-20210103153033519" style="zoom:67%;" />

![image-20210103153310737](tensorboard-no-module-past.assets/image-20210103153310737.png)

#### 安装注意事项
pip install tensorboard的时候会报错︰
ModuleNotFoundError: No module named 'past'
通过pip install future解决

#### TensorBoard

**SummaryWriter**
功能:提供创建event file的高级接口
主要属性:
log dir: event file输出文件夹
comment: 不指定log_dir时，文件夹后缀
filename suffix: event file文件名后缀

```python
class SummaryWriter(object):
def __init __(self, log_dir=None, comment='', purge_step=None, max_queue = 10, flush_secs=120, filename_suffix='')
```

#### 0 SummaryWriter

![image-20210103121042443](tensorboard-no-module-past.assets/image-20210103121042443.png)

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210103121111115.png" alt="image-20210103121111115" style="zoom: 67%;" />

设置log_dir，comment不起作用

![image-20210103160302097](tensorboard-no-module-past.assets/image-20210103160302097.png)

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210103160319449.png" alt="image-20210103160319449" style="zoom:67%;" />

**1. add_scalar()**
功能:记录标量
tag: 图像的标签名,图的唯一标识
scalar_value: 要记录的标量
global_step: x轴

**2. add_scalars()**
main_tag: 该图的标签
tag_scalar_dict : key是变量的tag, value是变量的值

```python
add_scalar(tag, scalar_value, global_step=None, walltime=None)
add_scalars(main_tag, tag_scalar_dict, global_step=None, walltime=None)
```

![image-20210103121208765](tensorboard-no-module-past.assets/image-20210103121208765-1609665008096.png)

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210103121306155.png" alt="image-20210103121306155" style="zoom:67%;" />

![image-20210103121544572](tensorboard-no-module-past.assets/image-20210103121544572-1609665008097.png)

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210103161339794.png" alt="image-20210103161339794" style="zoom:50%;" />

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210103161612514.png" alt="image-20210103161612514" style="zoom:67%;" />

**3. add_histogram()**

功能:统计直方图与多分位数折线图
tag: 图像的标签名，图的唯一标识
values: 要统计的参数
global_step: y轴
bins: 取直方图的bins

```python
add_histogram(tag, values, global_step=None, bins='tensorflow', walltime=None)
```

**2. histogram **

![image-20210103162314244](tensorboard-no-module-past.assets/image-20210103162314244.png)

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210103162350242.png" alt="image-20210103162350242" style="zoom:67%;" />

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210103162446219.png" alt="image-20210103162446219" style="zoom:50%;" />

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210103162422739.png" alt="image-20210103162422739" style="zoom:50%;" />

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210103162615843.png" alt="image-20210103162615843" style="zoom: 67%;" />

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210103162829358.png" alt="image-20210103162829358" style="zoom: 50%;" />

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210103162906885.png" alt="image-20210103162906885" style="zoom:50%;" />

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210103163059158.png" alt="image-20210103163059158" style="zoom:67%;" />

#### example

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210103164043282.png" alt="image-20210103164043282" style="zoom: 67%;" />

![image-20210103164126705](tensorboard-no-module-past.assets/image-20210103164126705.png)

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210103164323533.png" alt="image-20210103164323533" style="zoom: 67%;" />

#### acc & loss

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210103165311072.png" alt="image-20210103165311072" style="zoom: 50%;" />

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210103165405458.png" alt="image-20210103165405458" style="zoom: 50%;" />

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210103165220029.png" alt="image-20210103165220029" style="zoom:50%;" />

#### histogram

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210103170039104.png" alt="image-20210103170039104" style="zoom:50%;" />

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210103170107623.png" alt="image-20210103170107623" style="zoom:50%;" />

#### fc3.weight_data

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210103170241543.png" alt="image-20210103170241543" style="zoom:50%;" />

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210103170351288.png" alt="image-20210103170351288" style="zoom:50%;" />