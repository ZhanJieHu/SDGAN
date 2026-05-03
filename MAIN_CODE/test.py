import os
import argparse
import os
import torch
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

from model_all.data import Umake_data_loader
from model_all.engine.inference import inference
from model_all import build_model
from model_all.utils.checkpoint import MmnCheckpointer
from model_all.utils.comm import synchronize, get_rank
from model_all.utils.logger import setup_logger
from UTiLs import ConfigManager, move_all_tensors_to_device
import random
import numpy as np

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




def load_model_and_checkpoint(cfg, args):
    """加载模型和checkpoint"""
    # 构建模型
    model = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    move_all_tensors_to_device(model, device)
    model.eval()
    
    # 加载checkpoint
    output_dir = cfg.OUTPUT_DIR
    checkpointer = MmnCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(args.ckpt, use_latest=args.ckpt is None)
    
    return model


def run_inference(cfg, model, distributed):

    # load data
    dataset_names = cfg.DATASETS.NAME
    data_loaders_test = Umake_data_loader(cfg, mode="test", is_distributed=distributed)
    
    _ = inference(
        cfg,
        model,
        data_loaders_test,
        dataset_name=dataset_names,
        nms_thresh=cfg.TEST.NMS_THRESH,
        device=cfg.MODEL.DEVICE,
        epoch=-1, # 这里epoch参数没有实际意义，可以设置为任意值
    )
    synchronize()


def main():
    parser = argparse.ArgumentParser(description="DTFNet Inference")
    parser.add_argument(
        "--config-file",
        default="activity/text_new_230919/text_1_3w/config.yml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=0
    )
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device to use, e.g. 'cuda:0', 'cuda:1', 'cpu'",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    
    args = parser.parse_args()
    set_seed(42)

    # Distributed Training Settings
    num_gpus = int(os.environ.get("WORLD_SIZE", 1))
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    # load config
    cfg = ConfigManager.load_config(args.config_file, args)
    
    # validate config
    ConfigManager.validate_config(cfg)
    
    # setup logger
    save_dir = ""
    logger = setup_logger("dtf", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Checkpoint: {}".format(args.ckpt if args.ckpt else "latest"))
    logger.info(cfg)
    
    # load model and checkpoint
    model = load_model_and_checkpoint(cfg, args)
    
    # run inference
    run_inference(cfg, model, distributed)


if __name__ == "__main__":
    # If no arguments are passed
    import sys
    if not sys.argv[1:]:  # 如果没有传入参数
        sys.argv = [
            'test_net.py',
            '--config-file', '/media/zhangbolin/hu/OSGs/open_source/checkpoints/charades/100.open_source_test/config.yml',
            '--ckpt', '/media/zhangbolin/hu/OSGs/open_source/checkpoints/charades/100.open_source_test/model_20e.pth',
            '--device', 'cuda:1'
        ]
    
    main()