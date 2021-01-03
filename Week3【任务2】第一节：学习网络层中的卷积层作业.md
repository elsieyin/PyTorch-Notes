**Week3【任务2】第一节：学习网络层中的卷积层**

**作业名称（详解）：**

\1. 深入理解二维卷积，采用手算的方式实现以下卷积操作，然后用代码验证。

1）采用2个尺寸为3×3的卷积核对3通道的5×5图像进行卷积，padding=0， stride=1，dilation=0

其中 input shape = （3， 5， 5），数据如下

  ![img](http://wechatapppro-1252524126.file.myqcloud.com/image/ueditor/59300600_1585908227.png%21thumbnail)    

 

kernel size = 3*3， 第一个卷积核所有权值均为1， 第二个卷积核所有权值均为2，

计算输出的feature map尺寸以及所有像素值

2）接1）题，上下左右四条边均采用padding，padding=1，填充值为0，计算输出的feature map尺寸以及所有像素值

**打卡要求：代码输出结果截图，打印输出feature map的值**

\2. 对lena图进行3×3×33d卷积，提示：padding=（1， 0， 0）

```python
# ================ 3d
# flag = 1
flag = 0
if flag:
  conv_layer = nn.Conv3d(3, 1, (1, 3, 3), padding=(1, 0, 0))
  nn.init.xavier_normal_(conv_layer.weight.data)
  # calculation
  img_tensor.unsqueeze_(dim=2)  # B*C*H*W to B*C*D*H*W
  img_conv = conv_layer(img_tensor)
```

#### 

<img src="Week3%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%BD%91%E7%BB%9C%E5%B1%82%E4%B8%AD%E7%9A%84%E5%8D%B7%E7%A7%AF%E5%B1%82.assets/image-20201215134134288.png" alt="image-20201215134134288" style="zoom: 67%;" />

<img src="Week3%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%BD%91%E7%BB%9C%E5%B1%82%E4%B8%AD%E7%9A%84%E5%8D%B7%E7%A7%AF%E5%B1%82.assets/image-20201215134308526.png" alt="image-20201215134308526" style="zoom:80%;" />

<img src="Week3%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%BD%91%E7%BB%9C%E5%B1%82%E4%B8%AD%E7%9A%84%E5%8D%B7%E7%A7%AF%E5%B1%82.assets/image-20201215134543548.png" alt="image-20201215134543548" style="zoom: 67%;" />

#### 

![image-20201215144755846](Week3%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%BD%91%E7%BB%9C%E5%B1%82%E4%B8%AD%E7%9A%84%E5%8D%B7%E7%A7%AF%E5%B1%82%E4%BD%9C%E4%B8%9A.assets/image-20201215144755846.png)

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20201215145215611.png" alt="image-20201215145215611" style="zoom: 33%;" />



![image-20201215145412417](Week3%E3%80%90%E4%BB%BB%E5%8A%A12%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%BD%91%E7%BB%9C%E5%B1%82%E4%B8%AD%E7%9A%84%E5%8D%B7%E7%A7%AF%E5%B1%82%E4%BD%9C%E4%B8%9A.assets/image-20201215145412417.png)