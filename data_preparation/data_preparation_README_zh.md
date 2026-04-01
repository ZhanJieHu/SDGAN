# 动态特征和查询特征
本代码受 [UniSDNet](https://github.com/xian-sh/UniSDNet)的启发。我们沿用了 UniSDNet 所使用的动态特征（dynamic feature）与文本查询特征（query feature）。
动态特征和文本查询特征链接：
https://pan.baidu.com/s/1ktETHkIEliBEODfzCgl5vQ?pwd=5bwp
https://rochester.app.box.com/s/8znalh6y5e82oml2lr7to8s6ntab6mav/folder/137471786054

> **注意:**
> 1.此处我们只是提供了[UniSDNet](https://github.com/xian-sh/UniSDNet)已经公开的连接。在此感谢原作者们出色的开源贡献。
> 2.原作者提供了文本查询特征和音频查询特征。本项目只使用文本查询特征。
> 3.如果连接失效，可以选择[联系本项目作者](../README_zh.md#联系作者)

# 静态特征

静态特征连接：
Static feature download link: 
https://dx.doi.org/10.21227/t4rt-v882
```bibtex
@data{t4rt-v882-26,
doi = {10.21227/t4rt-v882},
url = {https://dx.doi.org/10.21227/t4rt-v882},
author = {Zhanjie Hu},
publisher = {IEEE Dataport},
title = {Static Feature of ActivityNet Caption, Charades-STA, and TACoS},
year = {2026} }
```

## 准备方法：
### 原视频下载
#### ActivityNet
[huggingface](https://huggingface.co/datasets/YimuWang/ActivityNet/tree/main)
可参考代码：./aNet_hf_video_download.py

[知乎](https://zhuanlan.zhihu.com/p/470416987)

#### Charades
[官网链接](https://prior.allenai.org/projects/charades)

#### TACoS
[官网链接](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/tacos-multi-level-corpus)


## 特征提取
静态特征的提取方法参考了 [OSDNet](https://github.com/Yisen-Feng/OSGNet/issues/4)。
请参考[StaticFeature](./StaticFeature/README.md)