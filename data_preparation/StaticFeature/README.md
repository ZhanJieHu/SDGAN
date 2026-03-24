## Environment

See the environment configuration file:
[enviroment](./0.env.txt)

---

## Pipeline

### 1. Video to Frames
Extract frames from raw videos:
[video2frames](./1.video2frames.py)

### 2. Frame Downsampling
Reduce the number of frames:
[frame_downsample](./2.frame_downsample.py)

---

## 3.CLIP Setup

Due to path dependencies, this project uses a **manually copied CLIP implementation**.

Original CLIP code:
[clip_model](./clip_model/)

### 3.1 Reproducing the Original Code

To reproduce the original implementation, you **must modify** the following file:

```text
./clip_model/clip.py
```

Specifically, update line 28:

```python
def _download(url: str, root: str = os.path.expanduser("/model_file_path")):
```

> Replace `"/model_file_path"` with your local directory for storing CLIP model weights.

---

### 3.2 Prepare CLIP Model

Download and prepare the CLIP model:
[prepare_clip_model](./3.prepare_clip_model.py)

---

## 4.Feature Extraction

Extract static features from frames:
[frame2feature](./4.frame2feature.py)



## 5.Recommended Alternative (Easier Setup)

You may install CLIP via pip instead:

```bash
pip install open_clip_torch
pip install git+https://github.com/openai/CLIP.git
