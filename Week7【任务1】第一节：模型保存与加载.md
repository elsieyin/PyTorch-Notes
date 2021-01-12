Week7【任务1】第一节：模型保存与加载

![image-20210108164616165](Week7%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E6%A8%A1%E5%9E%8B%E4%BF%9D%E5%AD%98%E4%B8%8E%E5%8A%A0%E8%BD%BD.assets/image-20210108164616165.png)

1. torch.save
  主要参数:
  obj：对象
  f：输出路径

2. torch.load
  主要参数
  f ：文件路径
  map location：指定存放位置，cpu or gpu

  #### 模型保存与加载

  #### 法1：保存整个Module
  torch.save(net, path)
  #### 法2：保存模型参数（推荐）
  state_dict = net.state_dict()
  torch.save(state_dict , path)

  <img src="Week7%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E6%A8%A1%E5%9E%8B%E4%BF%9D%E5%AD%98%E4%B8%8E%E5%8A%A0%E8%BD%BD.assets/image-20210108175211252.png" alt="image-20210108175211252" style="zoom: 50%;" />

  <img src="Week7%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E6%A8%A1%E5%9E%8B%E4%BF%9D%E5%AD%98%E4%B8%8E%E5%8A%A0%E8%BD%BD.assets/image-20210108175310294.png" alt="image-20210108175310294" style="zoom: 50%;" />

<img src="Week7%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E6%A8%A1%E5%9E%8B%E4%BF%9D%E5%AD%98%E4%B8%8E%E5%8A%A0%E8%BD%BD.assets/image-20210108180323704.png" alt="image-20210108180323704" style="zoom: 60%;" />

#### 断点续训练

```python
checkpoint = {
    "model_state_dict ": net.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch
}
```

![image-20210108181217642](Week7%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E6%A8%A1%E5%9E%8B%E4%BF%9D%E5%AD%98%E4%B8%8E%E5%8A%A0%E8%BD%BD.assets/image-20210108181217642.png)

![image-20210108181259512](Week7%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E6%A8%A1%E5%9E%8B%E4%BF%9D%E5%AD%98%E4%B8%8E%E5%8A%A0%E8%BD%BD.assets/image-20210108181259512.png)

![image-20210108181340289](Week7%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%B8%80%E8%8A%82%EF%BC%9A%E6%A8%A1%E5%9E%8B%E4%BF%9D%E5%AD%98%E4%B8%8E%E5%8A%A0%E8%BD%BD.assets/image-20210108181340289.png)