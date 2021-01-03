下划线 _ 是个变量，它指向“上一个”变量。你打开Python Shell，随手敲几个变量，然后print(_)观察就理解了。

<img src="file:///D:\Msg\qq\3204821761\Image\Group2\{}\3Z\{}3ZAMPF4WN4_89}N3M_8PG.png" alt="img" style="zoom: 67%;" />

Python支持的上下文管理

简单说，就是with语句所囊括的代码块
![img](file:///D:\Msg\qq\3204821761\Image\Group2\OK\7Q\OK7QM21WE@[J}I~T9T@_%RF.png)

红框里的代码就是该with语句囊括的代码块。

with的作用就是，执行红色代码的时候，首先获得一个上下文对象（通过_enable_get_lr_call(self)这个函数调用得到），然后执行上下文对象中的 __enter__() 里的内容，然后执行红色部分，然后执行上下文对象里__exit__() 里的内容。

https://docs.python.org/3/reference/compound_stmts.html#the-with-statement

简言之，就是执行某代码块，你想在执行它之前做点准备工作，在执行完它之后做点收尾工作。而准备工作和收尾工作这种杂活儿，不想一股脑写在主逻辑里，就可以用with语句。Do you get？