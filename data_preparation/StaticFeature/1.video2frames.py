# 遍历一个文件夹中的视频文件，并按固定间隔抽帧，将帧保存为图片。
import sys, os

# 设置根目录在 PYTHONPATH 里，根据实际情况而定
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))) 

#展示以及存储标注图像代码，
import os, tqdm, torch
import numpy as np
import cv2
import math


def get_certain_frame(model, video_path, save_dir, threshold=0.2, frame_interval=16):
    """
    从视频中按 frame_interval 抽帧并保存
    Args:
        model: 检测模型
        video_path: 输入视频路径
        save_dir: 保存帧的文件夹
        threshold: 置信度阈值（目前没用到，可以扩展）
        frame_interval: 抽帧间隔
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = math.ceil(total_frames / frame_interval)

    os.makedirs(save_dir, exist_ok=True)
    saved_frames = []

    idx = 0
    save_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            # 这里改为使用视频中的绝对帧索引命名
            save_path = os.path.join(save_dir, f"frame_{idx}.jpg")
            cv2.imwrite(save_path, frame)
            saved_frames.append(save_path)
            save_idx += 1
        idx += 1

    cap.release()
    return saved_frames


def get_videos_in_directory(folder_path):
    files = []
    for file_name in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.mpg', '.webm'}
            _, extension = os.path.splitext(file_name)
            if extension.lower() in video_extensions:
                files.append(file_name)
    return files


def get_files_in_directory(folder_path):
    files = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            files.append(file_name)
    return files


def main(args):
    video_path = args.video_path
    save_path = args.save_path
    
    video_names = get_videos_in_directory(video_path)
    save_names = get_files_in_directory(save_path)

    # 已处理过的视频名（不含扩展名）
    save_vid = [os.path.splitext(name)[0] for name in save_names]
    video_names = [name for name in video_names if os.path.splitext(name)[0] not in save_vid]

    # 初始化模型（这里你需要换成自己对应的配置和 checkpoint）
    # config_file = 'configs/co_detr/co_detr_r50_8xb2_150e_coco.py'
    # checkpoint_file = 'checkpoints/co_detr.pth'
    # model = init_detector(config_file, checkpoint_file, device=f'cuda:{args.gpu}')
    model = None  # 如果不需要检测，就先置空

    for video_name in tqdm.tqdm(video_names):
        print(f"处理视频: {video_name}")
        vid = os.path.splitext(video_name)[0]
        full_path = os.path.join(video_path, video_name)
        save_dir = os.path.join(save_path, vid)  # 每个视频一个子目录
        os.makedirs(save_dir, exist_ok=True)

        saved_frames = get_certain_frame(model, full_path, save_dir)
        print(f"{video_name} 保存了 {len(saved_frames)} 帧")


import argparse
def parse_args(video_path=None, save_path=None, gpu_id=0):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=gpu_id)  # 改为字符串类型
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--video-path", type=str, default=video_path)
    parser.add_argument("--save-path", type=str, default=save_path)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    video_path = "/media/zhangbolin/hu/OSGs/frame2feature_test/videos/"
    save_path = "/media/zhangbolin/hu/OSGs/frame2feature_test/frames/"

    gpu_id = 5  
    args = parse_args(video_path=video_path, save_path=save_path, gpu_id=gpu_id)
    device = 'cuda:{}'.format(args.gpu)
    main(args)
