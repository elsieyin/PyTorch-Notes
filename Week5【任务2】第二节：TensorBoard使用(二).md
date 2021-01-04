Week5【任务2】第二节：TensorBoard使用(二)

看最后一层，如果最后一层的weight_grad，也是随epoch的增加而慢慢减小到0，说明loss非常小，模型训练良好。而，如果最后一层的weight_grad比较大、不是0，则说明梯度往回传时发生了梯度消失，需具体分析。

SummaryWriter
**4.add_image() **
功能:记录图像
tag:图像的标签名，图的唯一标识
img tensor: 图像数据，**「注意尺度」**
	cond1: 图片像素值都在[0~1]之间，则默认在此基础上×255来可视化;
	cond2: 图片像素一般为[0~255]， 因此如果有像素值有大于1的，则落在此范围，不作改动。
global_step: x轴
dataformats: 数据形式, CHW, HWC, HW

```python
add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')
```

![image-20210103204522985](Week5%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9ATensorBoard%E4%BD%BF%E7%94%A8(%E4%BA%8C).assets/image-20210103204522985.png)

step1.<img src="http://localhost:6006/data/plugin/images/individualImage?ts=1609677138.702473&run=runs%5CJan03_20-32-17_DESKTOP-HDNB53Gtest_your_comment&tag=fake_img&sample=0&index=0" alt="img" style="zoom:25%;" />

step1.<img src="Week5%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9ATensorBoard%E4%BD%BF%E7%94%A8(%E4%BA%8C).assets/step1.png" alt="step1" style="zoom:25%;" />

step2.<img src="Week5%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9ATensorBoard%E4%BD%BF%E7%94%A8(%E4%BA%8C).assets/step2.png" alt="step2" style="zoom:25%;" />

step3.<img src="Week5%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9ATensorBoard%E4%BD%BF%E7%94%A8(%E4%BA%8C).assets/step3.png" alt="step3" style="zoom:25%;" />

step4.<img src="Week5%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9ATensorBoard%E4%BD%BF%E7%94%A8(%E4%BA%8C).assets/step4.png" alt="step4" style="zoom:25%;" />

step5.<img src="Week5%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9ATensorBoard%E4%BD%BF%E7%94%A8(%E4%BA%8C).assets/step5.png" alt="step5" style="zoom:25%;" />

**torchvision.utils.make_grid **
功能: 制作网格图像
tensor: 图像数据，B×C×H×W 形式
nrow: 行数（列数自动计算)
padding: 图像间距(像素单位)
normalize: 是否将像素值标准化 
	这里的标准化不同于一般的概念。CV中图片像素一般为 [0~255] ， 因此如果我们的像素值是 [0-1] 的数了，将这项设置为True，就会把像素值映射到0-255之间；设置为False，则不变。总之，这里的标准化是针对视觉像素	正常范围而言的。
range: 标准化范围
	有时需舍弃一些 过大 或 过小 的像素。例如，一张图片像素值范围是[-1000, 2000]， 如果指定这里的标准化范围是 [-600, 500] , 那么就会先把图片像素值 截断 到这个指定区间，小于 -600 的统一 -600 表示，大于 500 的统一500表示。然后再进行标准化到0-255.
scale_each: 是否单张图维度标准化 （因为有的图像可能尺度都不一样，如果设置False，是从整个大张量上进行标准化）
pad_value: padding的像素值 （网格线的颜色，通常默认0，无色）

```python
make_grid(tensor, nrow=8, padding=2, normalize=False, range=None, scale_each=False,
pad_value=0)
```

![image-20210104161141725](Week5%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9ATensorBoard%E4%BD%BF%E7%94%A8(%E4%BA%8C).assets/image-20210104161141725.png)

add_image结合make_grid的使用方法还是比较实用的，可以对数据进行一个基本的审查，快速检查 训练数据样本之间是否有交叉，这些样本的标签是否正确。这样审查数据集会比较快。

![makegrid](Week5%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9ATensorBoard%E4%BD%BF%E7%94%A8(%E4%BA%8C).assets/makegrid.png)

normalize=False:![normalize_false](Week5%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9ATensorBoard%E4%BD%BF%E7%94%A8(%E4%BA%8C).assets/normalize_false.png)

#### alexnet中的可视化

![image-20210104162425489](Week5%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9ATensorBoard%E4%BD%BF%E7%94%A8(%E4%BA%8C).assets/image-20210104162425489.png)

<img src="Week5%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9ATensorBoard%E4%BD%BF%E7%94%A8(%E4%BA%8C).assets/image-20210104162447434.png" alt="image-20210104162447434" style="zoom: 50%;" />

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210104165640842.png" alt="image-20210104165640842" style="zoom:33%;" />

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210104165014814.png" alt="image-20210104165014814" style="zoom:50%;" />

![image-20210104162703551](Week5%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9ATensorBoard%E4%BD%BF%E7%94%A8(%E4%BA%8C).assets/image-20210104162703551.png)



<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210104174721095.png" alt="image-20210104174721095" style="zoom:50%;" />

**5.add_graph()**
功能: 可视化模型计算图
model: 模型，必须是 nn.Module
input_to_model: 输出给模型的数据
verbose: 是否打印计算图结构信息

```python
add_graph(input_to_model=None, verbose=False)
```

![image-20210104171142974](Week5%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9ATensorBoard%E4%BD%BF%E7%94%A8(%E4%BA%8C).assets/image-20210104171142974.png)

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210104170928468.png" alt="image-20210104170928468" style="zoom: 80%;" />

**torchsummary**
功能:查看模型信息,便于调试
model: pytorch模型
input_size: 模型输入size
batch_size: batch size
device: “cuda” or “cpu”

```python
summary(model, input_size, batch_size=-1, device="cuda")
```

**pip install torchsummary**

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210104171048533.png" alt="image-20210104171048533" style="zoom:67%;" />