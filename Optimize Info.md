# 优化记录

## 第一次优化 v0.1

将image_transform 函数第一个循环并行化

```
当前加速比 1.32x
```

发现将image_transform的主循环并行化后该函数的性能占比已经大幅下降，因此暂不做进一步优化

存在的可优化内容：SIMD加法器对z1~z6的加法运算优化

### v0.1.1
尝试在加法部分进行simd优化，但提升不明显

## 第二次优化 v0.2

根据Vtune性能图，sgemm函数占据了大量性能，因此尝试并行化

### v0.2.0
尝试将外部主体循环并行化，采用guided方法

### v0.2.1
根据flame graph，发现函数在读取C_tenser为0时耗费了大量时间，原因是for循环是先m后n，这样会使数据分布不连续
因此考虑先n后m，交换循环顺序

这时load函数时间占比 ***55.7% --> 35.8%***

***仍存在优化空间***

```
当前加速比 1.78x
```

### v0.2.2
现在对所有可总体并行化函数进行并行化，包括

```
    output_transform
    image_transform
    sgemm
    winograd_convolution
    filter_packing
    image_packing
    output_inpacking_store
```

目前仍是粗糙并行化，对多层循环，暂时不适合更进一步并行化

当前加速比
```
    5.19x
```

### v0.2.3
现在对四层以上的嵌套循环采用了两层并行化，对性能有一定提升
同时将大规模加法simd化，有一定作用

当前运算量：
```
    平均14GFlops
```
当前加速比
```
    平均5.37x 
```

## v0.2.4
发现主程序两层循环多次调用sgemm函数，sgemm函数的开销是很大的，因此可以考虑全并行化
这样并行化的开销还是小于线性执行sgemm函数开销的

通过Vtune可以看到，现在cpu核心的调度已经可以有20核的正常调用了
并且线程创建函数以及乘法函数的有效时间大大增加

当前运算量：
```
    平均19.5GFlops
```
当前加速比
```
    平均7.14x 
```

不过可惜的是，这样又增大了线程创建的开销



### 当下问题
    1.线程等待占据了大量时间

    2.内存访问耗时，尤其这个访问了三个很大的三维张量



## V0.3

### 问题1
线程创建过多，创建线程函数的等待时间过长

发现将多层循环全部并行化对线程创建函数等待时间有损害
但适当的多层化有一定帮助

所以从目前看来，线程的创建有点多了

### 问题2
内存访问优化

### 尝试方向
    1.重构整个函数，全部放在一个并行区内，但这样会减少代码的模块化

    2.提高内存访问效率

### v0.3.0

将simd指令交给openMP降低了内存的读取次数，约10%

### v0.3.1
注意到z0~z6的多次读取，因此将其存放于结构体内，提高存储的连续性
发现内存读取次数下降了32%，但缓存命中率翻倍了，最终运行时间相近

在内存带宽利用率方面，低带宽区域变大，平局带宽下降25%，高带宽峰值显著下降
因此这一方面有所改进

追求高CPU利用率只有在程序是计算受限而非内存受限时才是最有效的策略。最终目标是更高的性能，而非更高的CPU利用率。

#### v0.3.1.1
折中方案，将z6变为独立变量，这时LLC Miss变为0，但读取次数变多
带宽利用率分布接近,放弃

*** 问题 ***
：目前优化内存并没有对时间有显著提升，要思考以下问题
很可能的原因是，并行等待时间仍然占据大部分时间

### v0.3.1.2
尝试将filtertransform求和顺序交换，提高顺序读取速率，但效果不明显
不少性能参数甚至变差了

### v0.3.2
思路来自 https://zhuanlan.zhihu.com/p/438173915
重大突破：优化sgemm算法，全部分块，向量化，当前速度可达0.055197
但应该可以进一步优化，方法待定

发现：并行度下降，可以考虑如何优化

当前加速比：8.86x