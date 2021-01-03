Week5【任务2】第二节：TensorBoard使用(二)

SummaryWriter
**4.add_image() **
功能:记录图像
tag:图像的标签名，图的唯一标识
img tensor: 图像数据，注意尺度
global_step: x轴
dataformats: 数据形式, CHW, HWC, Hw

```python
add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')
```

![image-20210103204522985](Week5%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9ATensorBoard%E4%BD%BF%E7%94%A8(%E4%BA%8C).assets/image-20210103204522985.png)

step1.<img src="http://localhost:6006/data/plugin/images/individualImage?ts=1609677138.702473&run=runs%5CJan03_20-32-17_DESKTOP-HDNB53Gtest_your_comment&tag=fake_img&sample=0&index=0" alt="img" style="zoom:25%;" />



step4.<img src="Week5%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9ATensorBoard%E4%BD%BF%E7%94%A8(%E4%BA%8C).assets/individualImage" alt="img" style="zoom:25%;" />

step5.<img src="Week5%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9ATensorBoard%E4%BD%BF%E7%94%A8(%E4%BA%8C).assets/individualImage" alt="img" style="zoom:25%;" />



**torchvision.utils.make_grid **
功能: 制作网格图像
tensor: 图像数据，B×C×H×W 形式
nrow: 行数（列数自动计算)
padding: 图像间距(像素单位)
normalize: 是否将像素值标准化
range: 标准化范围
scale_each: 是否单张图维度标准化
pad_value: padding的像素值

```python
make_grid(tensor, nrow=8, padding=2, normalize=False, range=None, scale_each=False,
pad_value=0)
```

**5.add_graph()**
功能: 可视化模型计算图
model: 模型，必须是 nn.Module
input_to_model: 输出给模型的数据
verbose: 是否打印计算图结构信息

```python
add_graph(input_to_model=None, verbose=False)
```

**torchsummary**
功能:查看模型信息,便于调试
model: pytorch模型
input_size: 模型输入size
batch_size: batch size
device: “cuda” or “cpu”

```python
summary(model, input_size, batch_size=-1, device="cuda")
```

