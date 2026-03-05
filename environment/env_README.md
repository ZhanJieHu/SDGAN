Runtime Environment:
Python 3.10 + PyTorch 1.13.1 (CUDA 11.7) + VideoMamba (causal-conv1d/mamba) + PyG (torch_geometric and its dependent packages)

PyTorch 2.X is recommended for better compatibility. PyTorch 1.13.1 is used here due to "path dependence".

<summary><b>📕 目录</b></summary>

- [Example](#Example)
- [Bug 1](./debug1.md)
- [Version of Dependencies](./pip_list_example.txt)

---
## Example
### Prerequisites
- A stable internet connection (required for downloading dependencies and packages).
- Optional: If you're in a restricted network environment, configure an HTTP/SOCKS proxy in your terminal, or use a domestic mirror like Tsinghua University Open Source Mirror (TUNA) for faster package installation.
- Sufficient disk space: at least 10 GB free is recommended.

### CUDA Toolkit
This example assumes that you have already installed the official **CUDA Toolkit 11.8** (backward compatible with CUDA 11.7 projects) released by NVIDIA (installation guide: https://docs.nvidia.com/cuda/archive/11.8.0/cuda-installation-guide-linux/index.html).
- To **fully reproduce the complete SDGAN codebase** (including the use of VideoMamba), you must build the relevant modules from source. This requires installing the **full system-level CUDA Toolkit** (including the nvcc compiler and complete toolchain).
- However, if you only want to **reproduce the core algorithmic ideas of SDGAN**, you can replace VideoMamba with the official `mamba-ssm` package (`pip install mamba-ssm`). This avoids the need to build from source and eliminates the dependency on a system-wide CUDA Toolkit installation.
- To use mamba-ssm instead of VideoMamba, simply change "mode": "VideoMamba" to "mode": "mamba" in the config file.

#### Permanently Configure CUDA Environment Variables
```bash
# Permanently add the following lines to your ~/.bashrc file
# This ensures the variables are set automatically in every new terminal session
# Adjust the path based on your local environment and CUDA setup.
echo "export CUDA_HOME=/usr/local/cuda-11.8" >> ~/.bashrc
echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> ~/.bashrc
echo "export CPATH=\$CUDA_HOME/include:\$CPATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc

# Reload the shell configuration to apply changes right away
source ~/.bashrc

# Quick checks to confirm everything is set correctly:
# Should print: /usr/local/cuda-11.8
echo $CUDA_HOME

# Should show something like: nvcc: NVIDIA (R) Cuda compiler driver ... release 11.8, ...
nvcc -V
```

#### Temporarily Configure CUDA Environment Variables
```bash
export CPATH=/usr/local/cuda-11.8/include:$CPATH 
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/include:$CPATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 1. Environment Preparation and Basic Configuration
#### 1.1 Create and Activate Conda Environment
```bash
# Create a Python 3.10 environment named [env_name]
conda create -n [env_name] python=3.10 -y
# Activate the environment
conda activate [env_name]
```

---

### 2、install PyTorch 1.13.1（CUDA 11.7）
```bash
# https://pytorch.org/get-started/previous-versions/
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
```

---

### 3 Install VideoMamba（causal-conv1d + mamba）
#### 3.1 Clone the Repository
```bash
# Clone the VideoMamba repository
git clone https://github.com/OpenGVLab/VideoMamba.git
```

#### 3.2 Modify the Build Configuration Files
```bash
# create and edit the files
vim [/path_to_VideoMamba]/causal-conv1d/pyproject.toml
vim [/path_to_VideoMamba]/mamba/pyproject.toml

# Make sure each file contains the following content:
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


#### 3.3 proceed with the editable installation steps (-e)
```bash
# --no-build-isolation avoids isolating the build environment and ensures torch is available
pip install -e ./VideoMamba/causal-conv1d --no-build-isolation
pip install -e ./VideoMamba/mamba --no-build-isolation
```
If you can not install, [debug1.md](./debug1.md) file might help you.

---

### 4. Install PyG and Its Dependencies
[PyG Offical Website](https://pytorch-geometric.readthedocs.io/en/latest/)
#### 4.1 Install Basic Dependencies
```bash
pip install terminaltables lmdb pandas transformers==4.28.0 numpy==1.26.4
```

#### 4.2 Install PyG Core Repository
```bash
pip install torch_geometric
pip install --verbose torch_scatter 
pip install --verbose torch_sparse
```
If the above installation fails, you can use the following commands as an alternative.

Explanation:
Pip downloads the source tar.gz package for torch_scatter from the mirror. During the wheel metadata build phase, it needs to `import torch` to check the version. However, pip's build isolation creates a temporary isolated environment without torch installed, resulting in `ModuleNotFoundError: No module named 'torch'`.

By explicitly specifying the wheel version (2.1.1+pt113cu117) and using `-f` to point to the official pre-built wheel index, pip is forced to download the binary .whl file directly. This completely skips source compilation and the `import torch` step in setup.py, bypassing the isolation issue and allowing the installation to complete successfully.
```bash
pip install torch-scatter==2.1.1+pt113cu117 -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install torch_sparse -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install yacs h5py terminaltables tqdm librosa datasets matplotlib fvcore
pip install seaborn
```

#### 4.3 Check PyTorch
```bash
# Installing PyG may turn PyTorch into a CPU-based version. Verify that PyTorch is the GPU version.
python -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Version:', torch.version.cuda); print('GPU available:', torch.cuda.is_available())"
# Expected output：
# PyTorch Version: 1.13.1
# CUDA Version: 11.7
# GPU available: True
```

#### 4.4 Install other Dependencies
```bash
pip install yacs h5py tqdm librosa datasets matplotlib fvcore
```