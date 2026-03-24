
'''
下面是提取图片编号的代码，需要根据实际情况确定。
def sort_frame_filenames(filenames):
    def extract_number(filename):
        match = re.search(r'-(\d+)\.jpg$', filename) # for charades
        return int(match.group(1)) if match else -1
    
    return sorted(filenames, key=extract_number)
'''
import torch
import sys, os
# 设置根目录在 PYTHONPATH 里，根据实际情况而定
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))) 

# 保存标注文件
import os
import json,tqdm

def get_framefloders_in_directory(folder_path):
    '''folder_path的所有子目录中，其内容只有jpg的子目录'''
    framefolders = []
    # 遍历 folder_path 下的所有子目录
    for subfolder_name in sorted(os.listdir(folder_path)):
        subfolder_path = os.path.join(folder_path, subfolder_name)
        if os.path.isdir(subfolder_path):  # 必须是文件夹
            all_files = os.listdir(subfolder_path)
            if all_files:  # 子文件夹不为空
                # 检查是否所有文件都是 jpg
                if all(file.lower().endswith(".jpg") for file in all_files):
                    framefolders.append(subfolder_name)
    return framefolders


def get_files_in_directory(folder_path):
    files = []
    # 使用os.listdir()获取指定文件夹中的所有文件和文件夹
    for file_name in os.listdir(folder_path):
        # 使用os.path.join()构建文件的完整路径
        file_path = os.path.join(folder_path, file_name)
        # 检查路径是否为文件而不是文件夹
        if os.path.isfile(file_path):
                files.append(file_name)
    return files

def get_ffloders(frame_path, save_path) -> list:
    all_ffolder_names = get_framefloders_in_directory(frame_path) # len(all_ffolder_names) = 273

    save_names = get_files_in_directory(save_path)   
    save_id = [name.split(".")[0] for name in save_names]
    # print(len(save_names),save_names[0])

    # 全部
    print(len(all_ffolder_names))
    print(all_ffolder_names)

    # 已处理
    print(len(save_id))
    print(save_id)

    # 待处理
    ffloder_names = [name for name in all_ffolder_names if name.split(".")[0] not in save_id]
    print(len(ffloder_names))
    print(ffloder_names)
    return ffloder_names

import re

def sort_frame_filenames(filenames):
    def extract_number(filename):
        match = re.search(r'-(\d+)\.jpg$', filename) # for charades
        return int(match.group(1)) if match else -1
    
    return sorted(filenames, key=extract_number)


def get_sorted_frame_filenames(frames_folder):
    files = [
        f for f in os.listdir(frames_folder)
        if f.lower().endswith(".jpg")
    ]
    return sort_frame_filenames(files)

#导入clip模型
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize,Lambda
from PIL import Image, ImageOps
# n_px=224
def pad_to_square(img, desired_size):
    old_size = img.size
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img = img.resize(new_size, resample=Image.LANCZOS)
    delta_w = desired_size - new_size[0]
    delta_h = desired_size - new_size[1]
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    return ImageOps.expand(img, padding)

large_process= Compose([
        # Resize(n_px, interpolation=Image.BICUBIC),#整体拉伸至宽为224
        Lambda(lambda img: pad_to_square(img, 336)),
        # CenterCrop(n_px)]#中心裁剪
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

import clip_model as clip
from PIL import Image
import os
import torch

# 提取特征
def extract_clip_feature(clip_extractor, img_files, process, device="cuda"):
    image_features = []
    for img_file in img_files:
        img = Image.open(img_file).convert("RGB")  # 打开 jpg
        img = process(img).unsqueeze(0).to(device)
        with torch.no_grad():
            image_feature = clip_extractor.encode_image(img)
        image_features.append(image_feature)
    image_features = torch.cat(image_features)
    return image_features.cpu().numpy()

import pickle
def save_feature(features, folder_path, video_name): 
    clip_large_file_path = os.path.join(folder_path, f"{video_name}.pkl")
    with open(clip_large_file_path, 'wb') as file:
        pickle.dump(features, file)   # ✅ 这里改成 dump

def load_feature(pkl_file_path):
    with open(pkl_file_path, 'rb') as f:
        return pickle.load(f)
    
if __name__ == '__main__':
    # 环境：OSG_detr 或者OSG_detr_2

    # 要获取文件名的文件夹路径
    frame_path = "/media/zhangbolin/hu/OSGs/frame2feature_test/frames"
    save_path = "/media/zhangbolin/hu/OSGs/frame2feature_test/features"
    ffloder_names = get_ffloders(frame_path=frame_path, save_path=save_path)

    gpu_id = 6   # 选择第几块GPU，比如 0,1,2...
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # # 第一次下载模型时，如果网络不好，可以设置代理
    # import os
    # os.environ["http_proxy"] = "http://127.0.0.1:7890"
    # os.environ["https_proxy"] = "http://127.0.0.1:7890"
    # os.environ["all_proxy"] = "http://127.0.0.1:7890"

    # 加载模型
    clip_large_model="ViT-L/14@336px"
    clip_large_extractor, _ = clip.load(clip_large_model, device=device, jit=False)


    for i in range(len(ffloder_names)):
        ffloder_name = ffloder_names[i]
        print(f"++++++第{i}个文件夹：{ffloder_name}++++++")
        # 一个帧文件夹
        ffloder_path = os.path.join(frame_path, ffloder_name) 

        # 帧文件夹中的帧们
        sorted_files = [os.path.join(ffloder_path, file_name) for file_name in get_sorted_frame_filenames(ffloder_path)]

        # 提取特征
        features = extract_clip_feature(clip_large_extractor, sorted_files, large_process, device=device)
        print(features.shape)

        # 存储特征
        save_feature(features, save_path, ffloder_name)

        # # 读取特征
        # features_loaded = load_feature(pkl_file_path)
        # print(type(features_loaded), features_loaded.shape)





