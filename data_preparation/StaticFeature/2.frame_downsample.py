'''
将文件夹中所有视频的所有帧文件下采样

# 视频文件夹
frame_path = "/media/zhangbolin/hu/OSGs/tacos/frames"
$ ls [frame_path]
00SL4  00T4B  013SD # 视频文件夹

$ ls [frame_path]/00SL4
zhangbolin@nbu-cs-02:/media/zhangbolin/datasets/charades/frame24fps/00SL4$ ls
00SL4-000001.jpg  00SL4-000002.jpg  00SL4-000003.jpg  ...

tips：
sort_frame_filenames对Charades数据集做了适配
'''

import sys, os
# 设置根目录在 PYTHONPATH 里，根据实际情况而定
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

#保存标注文件
import os
import json, tqdm
def save_jsonl(data, filename):
    """data is a list"""
    with open(filename, "w") as f:
        
        f.write("\n".join([json.dumps(e) for e in data]))

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
        # 匹配 - 后面的数字（帧号）
        match = re.search(r'-(\d+)\.jpg$', filename)
        return int(match.group(1)) if match else -1
    
    return sorted(filenames, key=extract_number)


def get_sorted_frame_filenames(frames_folder):
    files = [
        f for f in os.listdir(frames_folder)
        if f.lower().endswith(".jpg")
    ]
    return sort_frame_filenames(files)

import shutil

def save_frame(save_path, ffloder_name, sorted_files, stride):
    """
    下采样并保存帧文件
    
    Args:
        save_path: 保存根目录
        ffloder_name: 视频文件夹名称
        sorted_files: 排序后的帧文件路径列表
        stride: 下采样步长
    """
    # 创建保存该视频的文件夹
    video_save_folder = os.path.join(save_path, ffloder_name)
    os.makedirs(video_save_folder, exist_ok=True)
    
    # 按stride下采样
    sampled_files = sorted_files[::stride]
    
    print(f"原始帧数: {len(sorted_files)}, 下采样后帧数: {len(sampled_files)}")
    
    # 复制选中的帧文件
    for src_file in tqdm.tqdm(sampled_files, desc=f"Copying {ffloder_name}"):
        filename = os.path.basename(src_file)
        dst_file = os.path.join(video_save_folder, filename)
        shutil.copy2(src_file, dst_file)
    
    print(f"已保存 {len(sampled_files)} 帧到 {video_save_folder}")

if __name__ == '__main__':
    # 环境：OSG_detr 或者OSG_detr_2

    # 要获取文件名的文件夹路径
    frame_path = "/media/zhangbolin/hu/OSGs/frame2feature_test/frames"
    save_path = "/media/zhangbolin/hu/OSGs/frame2feature_test/down_sampled"
    ffloder_names = get_ffloders(frame_path=frame_path, save_path=save_path)
    stride = 16  # 每隔多少帧提取一次特征



    for i in range(len(ffloder_names)):
        ffloder_name = ffloder_names[i]
        print(f"++++++第{i}个文件夹：{ffloder_name}++++++")
        # 一个帧文件夹
        ffloder_path = os.path.join(frame_path, ffloder_name) 

        # 帧文件夹中的帧们
        sorted_files = [os.path.join(ffloder_path, file_name) for file_name in get_sorted_frame_filenames(ffloder_path)]

        # 保留特定帧
        save_frame(save_path, ffloder_name, sorted_files, stride)