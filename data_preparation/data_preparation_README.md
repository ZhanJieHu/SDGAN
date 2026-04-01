# dynamic feature and query feature
This project is inspired by [UniSDNet](https://github.com/xian-sh/UniSDNet). We use the same dynamic feature and query feature as UniSDNet.

Links of dynamic feature and query feature:
https://pan.baidu.com/s/1ktETHkIEliBEODfzCgl5vQ?pwd=5bwp
https://rochester.app.box.com/s/8znalh6y5e82oml2lr7to8s6ntab6mav/folder/137471786054

> **Remarks:**
> 1. The links were provided by [UniSDNet](https://github.com/xian-sh/UniSDNet). We thank the authors for their excellent open-source contributions.
> 2. The authors provide both speech query features and text query features. In this work, we use the text query features, referred to as "query features".
> 3. If any links have expired, please [contact us](../README.md#Contact).

# Static Features

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



## Preparation

### Download Original Videos

#### ActivityNet
- [Hugging Face](https://huggingface.co/datasets/YimuWang/ActivityNet/tree/main)  
  Reference code: `./aNet_hf_video_download.py`

- [Zhihu Guide](https://zhuanlan.zhihu.com/p/470416987)

#### Charades
- [Official Website](https://prior.allenai.org/projects/charades)

#### TACoS
- [Official Website](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/tacos-multi-level-corpus)

## Feature Extraction

The extraction method for static features is based on [OSDNet](https://github.com/Yisen-Feng/OSGNet/issues/4).

Please refer to [StaticFeature](./StaticFeature/README.md) for detailed steps.