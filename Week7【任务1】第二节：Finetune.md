Week7【任务1】第二节：Finetune

Transfer Learning:机器学习分支，研究源域(source domain)的知识如何应用到目标域(target domain)

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210111215537580.png" alt="image-20210111215537580" style="zoom: 50%;" />

《A Survey on Transfer Learning》

Model Finetune ：模型的迁移学习

<img src="Week7%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9AFinetune.assets/image-20210112110840940.png" alt="image-20210112110840940" style="zoom:56%;" />

<img src="Week7%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9AFinetune.assets/image-20210111215628745.png" alt="image-20210111215628745" style="zoom: 40%;" />

《How transfer able are features in deep neural networks?》

模型微调步骤:
1．获取预训练模型参数
2．加载模型(load_state_dict)
3．修改输出层

模型微调训练方法:
1.固定预训练的参数(requires_grad =False; Ir=O)
2.Features Extractor较小学习率(params_group)

Finetune Resnet-18用于二分类
蚂蚁蜜蜂二分类数据
训练集:各120～张
验证集:各70～张

数据https://download.pytorch.org/hymenoptera_data.zip
模型https://download.pytorch.org/models/resnet18-5c106cde.pth'

hellopytorch/data/ 

![image-20210112114700334](Week7%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9AFinetune.assets/image-20210112114700334.png)

![image-20210112114729272](Week7%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9AFinetune.assets/image-20210112114729272.png)

![image-20210112114751905](Week7%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9AFinetune.assets/image-20210112114751905.png)

![image-20210112114855433](Week7%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9AFinetune.assets/image-20210112114855433.png)

![image-20210112114915450](Week7%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9AFinetune.assets/image-20210112114915450.png)

![image-20210112115000472](Week7%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9AFinetune.assets/image-20210112115000472.png)

#### my_dataset.py

<img src="C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210112200812940.png" alt="image-20210112200812940" style="zoom:50%;" />

![image-20210112200827467](Week7%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9AFinetune.assets/image-20210112200827467.png)

![image-20210112200908387](Week7%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9AFinetune.assets/image-20210112200908387.png)