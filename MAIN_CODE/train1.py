import os
import argparse
import re
import glob
import random
import torch
from torch import optim
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
from model_all.data import Umake_data_loader
# from dtfnet.config import cfg
from model_all import Trainer, unpack_args_device, ddp_setup
from model_all.engine.inference import inference
from model_all.engine.trainer import do_train
from model_all import build_model
from model_all.utils.checkpoint import MmnCheckpointer
from model_all.utils import SDGANCheckpointer
from model_all.utils.comm import synchronize, get_rank
#from dtfnet.utils.imports import import_file
from model_all.utils.logger import setup_logger, setup_logger_time
from model_all.utils.miscellaneous import mkdir, save_config, save_config_unicode, save_config_as_json
import logging
from UTiLs import ConfigManager, move_all_tensors_to_device, pobj, save_file
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
def build_optimizer(model, cfg):
    """构建优化器和调度器"""
    base_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.AdamW(
        base_params, 
        lr=cfg.SOLVER.LR, # Learning rate
        betas=(0.9, 0.99), 
        weight_decay=1e-4 # 之前都是1e-5，但可能有点小
    )
    # MultiStepLR：达到milestones指定epoch，学习率*gamma(0.1)衰减
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=cfg.SOLVER.MILESTONES, # 例如 [10,20,30]，到对应epoch lr*0.1
        gamma=0.1
    )
    return optimizer, scheduler

def setup_distributed_model(model, local_rank, distributed):
    """设置分布式训练"""
    if distributed:
        return DDP(model)
    return model

def is_valid_checkpoint(filepath, min_size_mb=1):
    """综合检查文件完整性"""
    # 检查文件大小
    try:
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        if size_mb < min_size_mb:
            print(f"警告: {filepath} 文件过小 ({size_mb:.2f} MB)")
            return False
    except Exception as e:
        print(f"警告: 无法读取文件大小 {filepath}: {e}")
        return False
    
    # 尝试加载验证
    try:
        checkpoint = torch.load(filepath, map_location=torch.device("cpu"))
        required_keys = ["model"]  # 可以根据需要添加其他必需键
        if not all(key in checkpoint for key in required_keys):
            print(f"警告: {filepath} 缺少必需的键")
            return False
        return True
    except Exception as e:
        print(f"警告: 无法加载 {filepath}: {e}")
        return False

# 提取epoch数
def extract_epoch(filename):
    match = re.search(r'model_(\d+)e\.pth', os.path.basename(filename))
    return int(match.group(1)) if match else 0

def load_checkpoint_for_resume(cfg, model, optimizer, scheduler, distributed):
    """加载checkpoint用于恢复训练"""
    weight_files = glob.glob(os.path.join(cfg.OUTPUT_DIR, "model_*e.pth"))

    valid_files = [f for f in weight_files if is_valid_checkpoint(f)]
    if not valid_files:
        raise FileNotFoundError(f"在 {cfg.OUTPUT_DIR} 中找不到有效的完整权重文件")
    
    latest_epoch = max(extract_epoch(f) for f in valid_files)
    latest_weight_file = os.path.join(cfg.OUTPUT_DIR, f"model_{latest_epoch}e.pth")
    
    print(f"加载checkpoint: epoch={latest_epoch}, 文件={latest_weight_file}")
    
    checkpoint = torch.load(latest_weight_file, map_location=torch.device("cpu"))
    
    # 加载模型权重
    model_to_load = model.module if distributed else model
    model_to_load.load_state_dict(checkpoint["model"])
    
    # 加载优化器和调度器状态
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    
    return latest_epoch + 1

def train(cfg, local_rank, distributed):
    # build model
    model = build_model(cfg)
    
    device = torch.device(cfg.MODEL.DEVICE)
    move_all_tensors_to_device(model, device)
    # model = setup_distributed_model(model, local_rank, distributed)
    
    # build optimizer and scheduler
    optimizer, scheduler = build_optimizer(model, cfg)
    
    # prepare data loaders
    data_loader = Umake_data_loader(cfg, mode="train", is_distributed=distributed)
    synchronize()

    data_loader_val = None
    if cfg.SOLVER.VAL_PERIOD > 0:
        data_loader_val = Umake_data_loader(cfg, mode="val", is_distributed=distributed)
        synchronize()


    # set checkpointer
    save_to_disk = (get_rank() == 0)
    checkpointer = SDGANCheckpointer(
        cfg, model, optimizer, scheduler, 
        save_dir=cfg.OUTPUT_DIR,
        save_to_disk=save_to_disk,
        logger=logging.getLogger(__name__)
    )


    # get starting epoch for resume
    epoch_start = 1
    if cfg.SOLVER.RESUME:
        try:
            epoch_start = checkpointer.resume_from_checkpoint( # 使用新的resume_from_checkpoint方法
                checkpoint_pattern="model_*e.pth"
            )
            if local_rank == 0:
                print(f"从epoch {epoch_start} 继续训练")
        except Exception as e:
            if local_rank == 0:
                print(f"恢复训练失败: {e}，从头开始训练")
            epoch_start = 1

    # DDP 
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True  # 不强制参与前向传播的参数必须参与损失计算
        )
        checkpointer.model = model

    # start training
    do_train(
        cfg, model, data_loader, data_loader_val,
        optimizer, scheduler, checkpointer, device,
        cfg.SOLVER.CHECKPOINT_PERIOD, cfg.SOLVER.VAL_PERIOD,
        epoch_start=epoch_start,
        local_rank=local_rank if distributed else 0,
        distributed=distributed
    )
    
    return model

def run_test(cfg, model, distributed):
    if isinstance(model, DDP):
        model = model.module
    torch.cuda.empty_cache()
    dataset_name = cfg.DATASETS.NAME
    data_loader_val = Umake_data_loader(cfg, mode="test", is_distributed=distributed)
    inference(
        cfg,
        model,
        data_loader_val,
        dataset_name=dataset_name,
        nms_thresh=cfg.TEST.NMS_THRESH,
        device=cfg.MODEL.DEVICE,
        epoch=999,  # 这里epoch参数没有实际意义，可以设置为任意值
    )
    synchronize()

def set_seed(seed: int):
    # Set the seed for Python's built-in random module to ensure deterministic randomness
    random.seed(seed)
    # Set the seed for NumPy's random number generator (offset by 1 to avoid overlap with Python random)
    np.random.seed(seed + 1)
    # Set the seed for PyTorch's CPU random number generator (offset by 2 for separation)
    torch.manual_seed(seed + 2)
    # Set the seed for the default CUDA device's random number generator (offset by 4)
    torch.cuda.manual_seed(seed + 4)
    # Set the seed for ALL available CUDA devices (critical for multi-GPU training)
    torch.cuda.manual_seed_all(seed + 4)
    # code above refer to https://github.com/minghangz/cpl/blob/main/train.py#L24

    # recommended not to enable it. 
    # Once enabled, it cannot guarantee the bit-wise reproduction of results and will increase the computing time
    # # forces deterministic algorithms where possible
    # torch.backends.cudnn.deterministic = True
    # # avoids non-deterministic algorithm selection
    # torch.backends.cudnn.benchmark = False

    '''# cannot enable it: 
    RuntimeError: cumsum_cuda_kernel does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'. You can turn off determinism just for this operation, or you can use the 'warn_only=True' option, if that's acceptable for your application. You can also file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation.
    '''
    # torch.use_deterministic_algorithms(True)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Mutual Matching Network")
    parser.add_argument("--config-file", default="./configs/activitynet.yaml",
                       metavar="FILE", help="path to config file", type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--skip-test", dest="skip_test",
                       help="Do not test the final model", action="store_true")
    parser.add_argument("--device", default="cuda:0",
                       help="Device to use, e.g. 'cuda:0', 'cuda:1', 'cpu'", type=str)
    parser.add_argument("--tag", default="", help="决定存储文件夹", type=str)
    parser.add_argument("--resume", default="", help="继续训练的checkpoint目录路径", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line",
                       default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--distributed", default=False, type=bool)
    parser.add_argument("--num_gpus", default=1, type=int)
    args = parser.parse_args()

    # 3. 设置设备
    distributed, local_rank, gpu_id = setup_device_ddp(args.device)
    args.device = 'cuda:' + str(gpu_id)  # 更新为具体GPU设备字符串
    
    return args, distributed, local_rank

def setup_device_ddp(device_str):
    """设置设备并返回GPU ID"""
    mode, gpu_ids = unpack_args_device(device_str)
    
    distributed = False

    if mode == 'multi_gpu':
        print(f"===== 进程启动 =====")
        print(f"当前进程 PID: {os.getpid()}")
        print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'Not Set')}")
        print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'Not Set')}")
        print(f"可用GPU列表: {gpu_ids}")
        local_rank, gpu_id = ddp_setup(gpu_ids)
        print(f"[LOCAL_RANK {os.environ['LOCAL_RANK']}] Using GPU {gpu_id}")
        distributed = True

    elif mode == 'gpu':
        gpu_id = gpu_ids[0]
        local_rank = 0
    else:
        raise ValueError(f"未知的设备模式: {mode}")
    
    return distributed, local_rank, gpu_id

def load_or_resume_config(args):

    # config: 处理恢复训练的情况
    if args.resume:
        # 从checkpoint目录读取保存的配置文件
        checkpoint_dir = args.resume
        config_path = os.path.join(checkpoint_dir, 'config.yml')
        assert os.path.exists(config_path), f"找不到配置文件: {config_path}"
        
        # 加载checkpoint中的配置
        cfg = ConfigManager.load_config(config_path, args)

        # 设置为恢复模式
        cfg = ConfigManager.set_resume_mode(cfg, resume=True)

        # 使用checkpoint目录作为输出目录
        cfg = ConfigManager.set_checkpoint_dir(cfg, checkpoint_dir=checkpoint_dir)
        
    # config: 处理从头开始训练的情况
    else:
        # 加载用户指定的配置文件
        cfg = ConfigManager.load_config(args.config_file, args)

        # 设置为新训练模式
        cfg = ConfigManager.set_resume_mode(cfg, resume=False)

        # 根据配置和tag创建新的输出目录
        cfg = ConfigManager.set_checkpoint_dir(cfg, checkpoint_dir=cfg.OUTPUT_DIR, tag=args.tag)

    # 配置日志系统(恢复模式下也需要重新配置)
    ConfigManager.setup_logging(cfg, args)
    
    # 展开配置中的嵌套参数
    cfg = ConfigManager.expand_config(cfg)
    
    # 验证配置的有效性(使用前进行检查)
    ConfigManager.validate_config(cfg)

    return cfg

def record_cfg(cfg, args):

    # 保存完整配置到输出目录
    save_config_as_json(cfg, os.path.join(cfg.OUTPUT_DIR, 'config.yml'))

    train_file_path = os.path.abspath(__file__) # 当前文件的绝对路径
    ConfigManager.save_source_filesPro(output_dir=cfg.OUTPUT_DIR, config_file=args.config_file, train_file=train_file_path)        

def main():

    # Parsing sys.argv
    args, distributed, local_rank = parse_arguments()
    
    # set random seed
    set_seed(42)

    # load config
    cfg = load_or_resume_config(args)

    # record config and source files (only for new training, not for resume)
    if not args.resume: # 仅在新训练时保存
        if distributed: # 在分布式训练中仅rank 0保存
            if local_rank == 0:
                record_cfg(cfg, args)
        else:
            record_cfg(cfg, args)

    # train
    model = train(cfg, local_rank, distributed)
    synchronize()

    # inference
    if not args.skip_test:
        run_test(cfg, model, distributed)


if __name__ == "__main__":
    import sys
    if not sys.argv[1:]:  # Mode 1: train from scratch
        sys.argv = [
            'train_nocfg.py',
            '--config-file', '/media/zhangbolin/hu/OSGs/unet/config_baby.yaml', # path of config file
            '--device', 'cuda:2',
            '--tag', '100.test', # tag for output directory
        ]

    # if not sys.argv[1:]:  # Mode 2: resume training from checkpoint
    #     sys.argv = [
    #         'train1.py',
    #         '--resume', '/media/zhangbolin/hu/OSGs/open_source/应该就是没用/checkpoints/100.open_source', # path of checkpoint directory
    #         '--device', 'cuda:1',
    #     ]
    main()