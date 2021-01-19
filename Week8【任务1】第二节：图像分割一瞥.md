Week8【任务1】第二节：图像分割一瞥

![image-20210116131949576](C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210116131949576.png)

1.超像素分割：少量超像素代替大量像素，常用于图像预处理
2.语义分割：逐像素分类,无法区分个体
3.实例分割：对个体目标进行分割，像素级目标检测
4.全景分割：语义分割结合实例分割

![image-20210116132231632](Week8%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2%E4%B8%80%E7%9E%A5.assets/image-20210116132231632.png)

![image-20210116191254454](Week8%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2%E4%B8%80%E7%9E%A5.assets/image-20210116191254454.png)

模型如何完成图像分割？
答：图像分割由模型与人类配合完成
模型：将数据映射到特征
人类：定义特征的物理意义，解决实际问题

![image-20210116192235061](Week8%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2%E4%B8%80%E7%9E%A5.assets/image-20210116192235061.png)

#### PyTorch-Hub——PyTorch模型库，有大量模型供开发者调用

```python
1. torch.hub.load('pytorch/vision', 'deeplabv3_resnet101',pretrained=True)
model = torch.hub.load(github, model, *args, **kwargs)
功能：加载模型
主要参数：
•	github：str, 项目名，eg：pytorch/vision，<repo_owner/repo_name[:tag_name]>
•	model: str, 模型名
2.	torch.hub.list(github, force_reload=False)
3.	torch.hub.help(github, model, force_reload=False)
```

https://pytorch.org/hub

![image-20210116192434524](Week8%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2%E4%B8%80%E7%9E%A5.assets/image-20210116192434524.png)

![image-20210116192439675](Week8%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2%E4%B8%80%E7%9E%A5.assets/image-20210116192439675.png)

Ps：蓝色为小猫，绿色为小狗

深度学习中的图像分割模型
最主要贡献：
利用全卷积完成pixelwise prediction

《Fully Convolutional Networks for Semantic Segmentation》
![image-20210116192543932](Week8%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2%E4%B8%80%E7%9E%A5.assets/image-20210116192543932.png)

 《U-Net: Convolutional Networks for Biomedical Image Segmentation》

![image-20210116192622884](Week8%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2%E4%B8%80%E7%9E%A5.assets/image-20210116192622884.png)

![image-20210116192558412](Week8%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2%E4%B8%80%E7%9E%A5.assets/image-20210116192558412.png)

#### DeepLab系列——V1

主要特点：
1.孔洞卷积：借助孔洞卷积，增大感受野
2.CRF：采用CRF进行mask后处理

![image-20210116192711459](Week8%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2%E4%B8%80%E7%9E%A5.assets/image-20210116192711459.png)

《DeepLabv1 Semantic image segmentation with deep convolutional nets and fully connected CRFs》

#### DeepLab系列——V2

主要特点：
1. ASPP（Atrous spatial pyramid pooling ）：解决多尺度问题

![image-20210116192740396](Week8%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2%E4%B8%80%E7%9E%A5.assets/image-20210116192740396.png)

《DeepLab- Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs》

#### DeepLab系列——V3

![image-20210116192800859](Week8%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2%E4%B8%80%E7%9E%A5.assets/image-20210116192800859.png)

主要特点： 1. 孔洞卷积的串行 2. ASPP的并行
《DeepLabv3- Rethinking Atrous Convolution for Semantic Image Segmentation》

![image-20210116192830444](Week8%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2%E4%B8%80%E7%9E%A5.assets/image-20210116192830444.png)

![image-20210116192837236](Week8%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2%E4%B8%80%E7%9E%A5.assets/image-20210116192837236.png)

#### DeepLab系列-V3+

![image-20210116192910788](Week8%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2%E4%B8%80%E7%9E%A5.assets/image-20210116192910788.png)

主要特点： deeplabv3基础上加上Encoder-Decoder思想
《DeepLabv3- Rethinking Atrous Convolution for Semantic Image Segmentation》

《Deep Semantic Segmentation of Natural and Medical Images: A Review》2019

图像分割资源：
https://github.com/shawnbit/unet-family 
https://github.com/yassouali/pytorch_segmentation

![image-20210116192941756](Week8%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2%E4%B8%80%E7%9E%A5.assets/image-20210116192941756.png)

Unet实现人像抠图(Portrait Matting)

<img src="Week8%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2%E4%B8%80%E7%9E%A5.assets/image-20210116193015380.png" alt="image-20210116193015380" style="zoom:50%;" /><img src="Week8%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2%E4%B8%80%E7%9E%A5.assets/image-20210116193029178.png" alt="image-20210116193029178" style="zoom:50%;" />

![image-20210116193040500](Week8%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9A%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2%E4%B8%80%E7%9E%A5.assets/image-20210116193040500.png)

数据来源：https://github.com/PetroWu/AutoPortraitMatting