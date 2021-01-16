Week8【任务1】第一节：图像分类一瞥

#### 模型如何完成图像分类的

![image-20210114103350952](C:%5CUsers%5C86182%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210114103350952.png)

3-d 张量 → 字符串
1. 类别名与标签的转换
2. 取输出向量最大值的标号
3. 复杂运算

```python
label_name = {"ants": 0, "bees": 1}
_, predicted = torch.max(outputs.data, 1)
outputs = resnet18(img_tensor)
```

答: 图像分类由模型与人类配合完成
模型: 将数据映射到特征
人类: 定义特征的物理意义，解决实际问题
![image-20210114103757902](Week8%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E4%B8%80%E7%9E%A5.assets/image-20210114103757902.png)

图像分类的Inference(推理)步骤:
1. 获取数据与标签
2. 选择模型，损失函数，优化器
3. 写训练代码
4. 写inference代码

lnference代码基本步骤:
1. 获取数据与模型
2. 数据变换，如RGB  → 4D-Tensor
3. 前向传播
4. 输出保存预测结果

lnference阶段注意事项:

1. 确保model处于eval状态而非training
2. 设置torch.no_ grad()，减少内存消耗
3. 数据预处理需保持一致，RGB or BGR?

He K , Zhang X , Ren S , et al. Deep Residual Learning for Image Recognition

<img src="Week8%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E4%B8%80%E7%9E%A5.assets/image-20210114104225766.png" alt="image-20210114104225766" style="zoom: 50%;" />

![image-20210114104237944](Week8%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E4%B8%80%E7%9E%A5.assets/image-20210114104237944.png)