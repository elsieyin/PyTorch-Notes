Week6【任务1】第二节：weight_decay正则化之Dropout

Dropout：随机失活
随机：dropout probability
失活：weight = 0

《Dropout: A simple way to prevent neural networks from overfitting 》

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9Aweight_decay%E6%AD%A3%E5%88%99%E5%8C%96%E4%B9%8BDropout.assets/image-20210106174557120.png" alt="image-20210106174557120" style="zoom: 33%;" />

![image-20210106194551165](Week6%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9Aweight_decay%E6%AD%A3%E5%88%99%E5%8C%96%E4%B9%8BDropout.assets/image-20210106194551165.png)

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9Aweight_decay%E6%AD%A3%E5%88%99%E5%8C%96%E4%B9%8BDropout.assets/image-20210106192936799.png" alt="image-20210106192936799" style="zoom: 67%;" />

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9Aweight_decay%E6%AD%A3%E5%88%99%E5%8C%96%E4%B9%8BDropout.assets/image-20210106192949907.png" alt="image-20210106192949907" style="zoom: 67%;" />

![image-20210106193010622](Week6%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9Aweight_decay%E6%AD%A3%E5%88%99%E5%8C%96%E4%B9%8BDropout.assets/image-20210106193010622.png)

![image-20210106193030354](Week6%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9Aweight_decay%E6%AD%A3%E5%88%99%E5%8C%96%E4%B9%8BDropout.assets/image-20210106193030354.png)

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9Aweight_decay%E6%AD%A3%E5%88%99%E5%8C%96%E4%B9%8BDropout.assets/image-20210106193112872.png" alt="image-20210106193112872" style="zoom:67%;" />

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9Aweight_decay%E6%AD%A3%E5%88%99%E5%8C%96%E4%B9%8BDropout.assets/image-20210106193443295.png" alt="image-20210106193443295" style="zoom: 67%;" />

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9Aweight_decay%E6%AD%A3%E5%88%99%E5%8C%96%E4%B9%8BDropout.assets/image-20210106193509839.png" alt="image-20210106193509839" style="zoom:67%;" />

![image-20210106194629433](Week6%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9Aweight_decay%E6%AD%A3%E5%88%99%E5%8C%96%E4%B9%8BDropout.assets/image-20210106194629433.png)

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9Aweight_decay%E6%AD%A3%E5%88%99%E5%8C%96%E4%B9%8BDropout.assets/image-20210106193748703.png" alt="image-20210106193748703" style="zoom: 80%;" />

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9Aweight_decay%E6%AD%A3%E5%88%99%E5%8C%96%E4%B9%8BDropout.assets/image-20210106193847553.png" alt="image-20210106193847553" style="zoom:67%;" />

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9Aweight_decay%E6%AD%A3%E5%88%99%E5%8C%96%E4%B9%8BDropout.assets/image-20210106194054810.png" alt="image-20210106194054810" style="zoom: 60%;" />

数据尺度变化：测试时，所有权重乘以1-drop probdrop prob = 0.3, 1-drop prob = 0.7

`nn.Dropout`
功能：Dropout层
参数:
p︰被舍弃 概率，失活 概率
实现细节:
训练时权重均乘以$\frac{1}{1-p}$，即除以$1-p$

```python
torch.nn.Dropout(p=0.5, inplace = False)
```



<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9Aweight_decay%E6%AD%A3%E5%88%99%E5%8C%96%E4%B9%8BDropout.assets/image-20210106194310914.png" alt="image-20210106194310914" style="zoom:67%;" />

<img src="Week6%E3%80%90%E4%BB%BB%E5%8A%A11%E3%80%91%E7%AC%AC%E4%BA%8C%E8%8A%82%EF%BC%9Aweight_decay%E6%AD%A3%E5%88%99%E5%8C%96%E4%B9%8BDropout.assets/image-20210106194354087.png" alt="image-20210106194354087" style="zoom: 67%;" />