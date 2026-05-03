if __name__ == '__main__':

    # 要获取文件名的文件夹路径
    label_file = "/media/zhangbolin/hu/OSGs/unet/data/tacos/val_audio_chinese.json"
    
    import json
    try:
        with open(label_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功读取文件: {label_file}")
    except Exception as e:
        print(f"读取文件失败: {str(e)}")
        raise

    total_videos = len(data)
    print(f"总共有 {total_videos} 个视频")    
'''
(/media/zhangbolin/conda_envs/unet) zhangbolin@nbu-cs-02:/media/zhangbolin/hu/OSGs/unet (unet_plus)$ /media/zhangbolin/conda_envs/unet/bin/python /media/zhangbolin/hu/OSGs/unet/MY_utils/READ_LABEL/main.py
成功读取文件: /media/zhangbolin/hu/OSGs/tacos/chinese/Translation/labels/tacos/train_audio_chinese.json
总共有 75 个视频
(/media/zhangbolin/conda_envs/unet) zhangbolin@nbu-cs-02:/media/zhangbolin/hu/OSGs/unet (unet_plus)$ /media/zhangbolin/conda_envs/unet/bin/python /media/zhangbolin/hu/OSGs/unet/MY_utils/READ_LABEL/main.py
成功读取文件: /media/zhangbolin/hu/OSGs/unet/data/tacos/test_audio_chinese.json
总共有 25 个视频
(/media/zhangbolin/conda_envs/unet) zhangbolin@nbu-cs-02:/media/zhangbolin/hu/OSGs/unet (unet_plus)$ /media/zhangbolin/conda_envs/unet/bin/python /media/zhangbolin/hu/OSGs/unet/MY_utils/READ_LABEL/main.py
成功读取文件: /media/zhangbolin/hu/OSGs/unet/data/tacos/val_audio_chinese.json
总共有 27 个视频
'''