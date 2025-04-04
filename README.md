# 基本说明

## 性能结果
  程序最佳性能：
  
        vgg16:  2.98s， 752.25GFlops  加速比 = 423.6 / 2.98 = 142.15x   (见Best_Results_data/result-big5.out)
        small:  0.0363s  154.35GFlops  加速比 = 0.8534 / 0.0363 = 23.50x
  
  
  程序总体性能（vgg16）

        整体分布： 3.02 +- 0.04s   735 +- 17GFlops
        中位数：   3.00            745
  
  具体数值可能与集群当前状态有关，在短时间内的多次运行结果相对稳定。

## 程序运行
  1.使用make进行编译，额外库的参数已经修改，采用gcc编译器

  2.运行时使用 ***run.sh*** 运行winograd程序；默认使用vgg16测试集，可选参数：1--vgg16，0--small

  3.运行环境：在运行make前请先运行env.sh，加载环境，程序使用了cuda12.8.0

  ***注***：由于程序在本地编译时连接了libcudart.so.11.0库，这个库只有hepnode1有，节点2，3是12.0版本，如果程序在不在节点1运行，会出现如下报错

      ./winograd: error while loading shared libraries: libcudart.so.11.0: cannot open shared object file: No such file or directory

  不过run.sh中已经默认规定了只使用节点1运行程序，所以直接运行即可

## 程序基本说明
  1.程序主要通过CUDA库对矩阵计算进行加速，并对数据传输做了一定优化，同时尽可能提高并行度。

  2.程序目前存在一些多余代码（即命名成正式函数的子名称）是一些有望实现但目前存在困难的优化方法
    主要包括：

      packing和transform函数的分块化处理
      gpu多流处理以及cpu、gpu流水线计算

