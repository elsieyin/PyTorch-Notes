Week7【任务1】第二节：Finetune

Transfer Learning:机器学习分支，研究源域(source domain)的知识如何应用到目标域(target domain)

![image-20210111215537580](C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210111215537580.png)

《A Survey on Transfer Learning》

Model Finetune ：模型的迁移学习

![image-20210111215628745](Week7%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9AFinetune.assets/image-20210111215628745.png)

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