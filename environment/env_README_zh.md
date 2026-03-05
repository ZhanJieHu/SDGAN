使用环境：
Python 3.10 + PyTorch 1.13.1（CUDA 11.7） + VideoMamba（causal-conv1d/mamba） + PyG（torch_geometric 及配套依赖）
Pytorch 2.X会有更好的兼容性，由于路径依赖的原因使用PyTorch 1.13.1

<summary><b>📕 目录</b></summary>

- [安装示例](#安装示例)
- [故障1](./debug1.md)
- [依赖库的版本](./pip_list_example.txt)

---
## 安装示例
### 事先准备
确保网络畅通可以考虑在命令行配置端口代理。或者使用清华源。
同时保证足够大的硬盘空间，建议大于10G

### CUDA Toolkit
本案例假设您已安装 NVIDIA 官方发布的 **CUDA Toolkit 11.8**（安装指南：https://docs.nvidia.com/cuda/archive/11.8.0/cuda-installation-guide-linux/index.html）。
- 如果您希望**完整复现 SDGAN 的全部代码**（包括使用 VideoMamba），则需要从源代码构建相关模块，因此**必须安装系统级的 CUDA Toolkit**（nvcc 编译器等完整工具链）。
- 但如果您**仅希望复现 SDGAN 的核心算法思想**，可以使用官方的 `mamba-ssm` 包（pip install mamba-ssm）来替代 VideoMamba，从而避免从源代码构建和对系统 CUDA Toolkit 的依赖。
- 使用mamba-ssm替代VideoMamba，只需将config文件中"mode": "VideoMamba"改为"mode": "mamba"。
#### 永久配置CUDA环境变量
```bash
# 永久写入环境变量（避免每次激活都配置）
# 根据您的本地环境和CUDA设置调整路径。
echo "export CUDA_HOME=/usr/local/cuda-11.8" >> ~/.bashrc
echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> ~/.bashrc
echo "export CPATH=\$CUDA_HOME/include:\$CPATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
# 立即生效环境变量
source ~/.bashrc

# 验证CUDA路径（输出/usr/local/cuda-11.8即为正确）
echo $CUDA_HOME
# 验证nvcc版本（输出V11.8.x，向下兼容11.7）
nvcc -V
```

# 暂时配置CUDA环境变量
```bash
export CPATH=/usr/local/cuda-11.8/include:$CPATH 
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/include:$CPATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 一、环境准备与基础配置
#### 1. 创建并激活conda环境
```bash
# 创建Python 3.10环境，命名为[env_name]
conda create -n [env_name] python=3.10 -y
# 激活环境
conda activate [env_name]
```

---

### 二、安装PyTorch 1.13.1（CUDA 11.7）
```bash
# https://pytorch.org/get-started/previous-versions/
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
```

---

### 3安装VideoMamba（causal-conv1d + mamba）
#### 3.1 克隆代码仓
```bash
# 克隆VideoMamba仓库
git clone https://github.com/OpenGVLab/VideoMamba.git
```

#### 3.2 修改构建配置文件
```bash
# 在指定路径创建文件
vim [/path_to_VideoMamba]/causal-conv1d/pyproject.toml
vim [/path_to_VideoMamba]/mamba/pyproject.toml

# 确保每个文件的内容包含以下内容
([env_name])username@hostname:[/path_to_VideoMamba]$ cat VideoMamba/causal-conv1d/pyproject.toml
[build-system]
requires = ["setuptools>=61", "wheel", "torch"]
build-backend = "setuptools.build_meta"
([env_name])username@hostname:[/path_to_VideoMamba]$ cat VideoMamba/mamba/pyproject.toml
[build-system]
requires = ["setuptools>=61", "wheel", "torch"]
build-backend = "setuptools.build_meta"
([env_name])username@hostname:[/path_to_VideoMamba]$ 
```


#### 3.3 以开发模式安装（-e）
```bash
# 安装causal-conv1d（--no-build-isolation避免隔离构建环境，确保torch可用）
pip install -e ./VideoMamba/causal-conv1d --no-build-isolation
pip install -e ./VideoMamba/mamba --no-build-isolation
```
如果无法安装[debug1.md](./debug1.md)可能会帮助你.

---

### 4. 安装PyG及配套依赖
[PyG官网](https://pytorch-geometric.readthedocs.io/en/latest/)
#### 4.1 安装基础依赖
```bash
pip install terminaltables lmdb pandas transformers==4.28.0 numpy==1.26.4
```

#### 4.2 安装PyG核心库
```bash
pip install torch_geometric
pip install --verbose torch_scatter 
pip install --verbose torch_sparse
```
如果上面出现问题，可使用以下代码替代。

原理：pip 从镜像下载了 torch_scatter 的源代码包（tar.gz），在构建 wheel 的 metadata 阶段需要 import torch 来检查版本，但 pip 的 build isolation 机制让这个临时隔离环境里没有 torch，导致 ModuleNotFoundError: No module named 'torch'。

通过明确指定 wheel 版本号（2.1.1+pt113cu117）并结合 -f 指向官方 wheel 索引，强制 pip 直接下载已经预编译好的二进制 wheel 文件（.whl），完全跳过从源代码构建和 setup.py 中的 import torch 步骤，从而绕过了隔离环境找不到 torch 的问题，直接完成安装。
```bash
pip install torch-scatter==2.1.1+pt113cu117 -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install torch_sparse -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install yacs h5py terminaltables tqdm librosa datasets matplotlib fvcore
pip install seaborn
```

#### 4.3 检查PyTorch
安装PyG可能会让PyTorch变为CPU版。验证PyTorch是否为GPU版本。
```bash
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA版本:', torch.version.cuda); print('GPU可用:', torch.cuda.is_available())"
# 预期输出：
# PyTorch版本: 1.13.1
# CUDA版本: 11.7
# GPU可用: True
```

#### 4.4 安装其他依赖
```bash
pip install yacs h5py tqdm librosa datasets matplotlib fvcore
```