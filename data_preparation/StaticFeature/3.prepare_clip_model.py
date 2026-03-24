import clip_model as clip
from clip_model.clip import _MODELS
import torch

print("Optional Versions:", list(_MODELS.keys()))

# 第一次下载模型时，如果网络不好，可以设置代理
import os
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["all_proxy"] = "http://127.0.0.1:7890"

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_large_model="ViT-L/14@336px"

# 修改一行代码：
# def _download(url: str, root: str = os.path.expanduser("/root/autodl-tmp/model/clip")):
# 这行代码中的路径必须修改成你自己的路径

clip_large_extractor, _ = clip.load(clip_large_model, device=device, jit=False) # 第一次加载时自动下载模型


