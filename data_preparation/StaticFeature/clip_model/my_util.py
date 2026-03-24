#util
import json,os,tqdm,torch
from itertools import groupby
from feature_extraction import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize,Lambda
from PIL import Image, ImageOps
import torch
import numpy as np
def read_jsonl_file(file_path):
    # 创建一个空列表来存储解析后的JSON对象
    data = []

    # 打开文件并逐行读取
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 解析当前行的JSON内容并添加到数据列表中
            json_object = json.loads(line.strip())
            data.append(json_object)
    
    return data
#导入标注数据
object_query_train_path='/home/feng_yi_sen/GroundNLQ-DINO/ego4d_data/ego4d_nlq_train_v2_label_lemma.jsonl'
object_query_val_path='/home/feng_yi_sen/GroundNLQ-DINO/ego4d_data/ego4d_nlq_val_v2_label_lemma.jsonl'
object_query_test_path='/home/feng_yi_sen/GroundNLQ-DINO/ego4d_data/ego4d_nlq_test_v2_label_lemma.jsonl'
object_query_train=read_jsonl_file(object_query_train_path)
object_query_val=read_jsonl_file(object_query_val_path)
object_query_test=read_jsonl_file(object_query_test_path)
object_query=object_query_train+object_query_val+object_query_test
grouped_object_query={key: list(group) for key, group in groupby(object_query, key=lambda x: x['video_id'])}
#导入clip模型

n_px=224
def pad_to_square(img, desired_size):
    old_size = img.size
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img = img.resize(new_size, resample=Image.LANCZOS)
    delta_w = desired_size - new_size[0]
    delta_h = desired_size - new_size[1]
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    return ImageOps.expand(img, padding)
process= Compose([
        # Resize(n_px, interpolation=Image.BICUBIC),#整体拉伸至宽为224
        Lambda(lambda img: pad_to_square(img, n_px)),
        # CenterCrop(n_px)]#中心裁剪
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
large_process= Compose([
        # Resize(n_px, interpolation=Image.BICUBIC),#整体拉伸至宽为224
        Lambda(lambda img: pad_to_square(img, 336)),
        # CenterCrop(n_px)]#中心裁剪
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

clip_base_model = "ViT-B/32"
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_base_extractor, _ = clip.load(clip_base_model, device=device, jit=False)
clip_large_model="ViT-L/14@336px"
clip_large_extractor, _ = clip.load(clip_large_model, device=device, jit=False)
#确定阈值
def get_thres(class_id,grouped_video_anno,frame_min_num,score_thr_low):
    frame_scores=[]
    for video_anno in tqdm.tqdm(grouped_video_anno):
        scores=[]
        for video_a in video_anno:
            if class_id==video_a['class_id']:
                score=video_a['score']
                scores.append(score)
        if len(scores)!=0:
            frame_scores.append(max(scores))
    if len(frame_scores)<frame_min_num:
        thres=0
    else:
        frame_scores=sorted(frame_scores)
        thres=min(frame_scores[-frame_min_num],score_thr_low)
    return thres
#提取特征
def extract_clip_feature(clip_extractor,imgs,process):
    image_features=[]
    for img in imgs:
        img=Image.fromarray(img)
        # img.show()
        img=process(img).unsqueeze(0).to(device)
        with torch.no_grad():
            image_feature= clip_extractor.encode_image(img)
        image_features.append(image_feature)
    image_features = torch.cat(image_features)
    return image_features.cpu().numpy()
def extract_regions(image, bboxs):
    image=image[:,:,::-1]#需要转换为RGB
    regions = []
    for bbox in bboxs:
        bbox_int = [round(x) for x in bbox]
        region = image[bbox_int[1]:bbox_int[3], bbox_int[0]:bbox_int[2]]
        regions.append(region)
    return regions
import pickle
score_thr_low=0.4
score_thr_high=0.6
frame_min_num=10
clip_base_save_path='/root/autodl-tmp/data/ego4d/nlq/co-detr/clip-base'
clip_large_save_path='/root/autodl-tmp/data/ego4d/nlq/co-detr/clip-large'
hs_save_path='/root/autodl-tmp/data/ego4d/nlq/co-detr/hs'
def extract_feature(annos,vid,hs_np,frames):
    grouped_object_anno=[list(group) for key, group in groupby(annos, key=lambda x: x['frame_id'])]
    video_object_query=grouped_object_query[vid]
    for object_query in video_object_query:
        labels=object_query['label']
        for label in labels:#枚举每个类别
            clip_base_file_path=os.path.join(clip_base_save_path,'{}_{}.pkl'.format(vid,label['class_id']))
            if os.path.exists(clip_base_file_path):
                continue
            #确定阈值
            thres=get_thres(label,grouped_object_anno,frame_min_num,score_thr_low)
            #迭代每帧
            clip_base_all=[]
            clip_large_all=[]
            hs_all=[]
            for frame_object_anno in tqdm.tqdm(grouped_object_anno):
                
                if len(frame_object_anno)!=0:
                    fid=frame_object_anno[0]['frame_id']
                    # out_file='/feng_yi_sen/have_a_look/Co-DETR/results/co_dino_5scale_lsj_vit_large_lvis/visualize/{}/{}/{}/{}.jpg'.format(vid,label['class_name'],score_thr,fid//16)
                #获取每帧的阈值
                scores=[]
                for per_object_anno in frame_object_anno:#迭代每个object
                    
                    if label['class_id']==per_object_anno['class_id']:
                        score=per_object_anno['score']
                        if score>=thres:
                            scores.append(score)
                if len(scores)<=0:
                    clip_base_all.append(None)
                    clip_large_all.append(None)
                    hs_all.append(None)
                    continue
                thres_high=min(max(scores),score_thr_high)
                #获取regions
                boxes=[]
                frame_hs=[]
                for per_object_anno in frame_object_anno:#迭代每个object
                    if label['class_id']==per_object_anno['class_id']:
                        score=per_object_anno['score']
                        if score>=thres_high:  
                            boxes.append(per_object_anno['bbox']+[score])
                            frame_hs.append(hs_np[per_object_anno['object_id']])
                regions=extract_regions(frames[fid],boxes)
                clip_base_features=extract_clip_feature(clip_base_extractor,regions,process)
                clip_large_features=extract_clip_feature(clip_large_extractor,regions,large_process)
                clip_base_all.append(clip_base_features)
                clip_large_all.append(clip_large_features)
                frame_hs=np.array(frame_hs)
                hs_all.append(frame_hs)
            clip_base_value = pickle.dumps(clip_base_all)
            with open(clip_base_file_path, 'wb') as file:
                file.write(clip_base_value)
            clip_large_value = pickle.dumps(clip_large_all)
            clip_large_file_path=os.path.join(clip_large_save_path,'{}_{}.pkl'.format(vid,label['class_id']))
            with open(clip_large_file_path, 'wb') as file:
                file.write(clip_large_value)
            hs_value = pickle.dumps(hs_all)
            hs_file_path=os.path.join(hs_save_path,'{}_{}.pkl'.format(vid,label['class_id']))
            with open(hs_file_path, 'wb') as file:
                file.write(hs_value)