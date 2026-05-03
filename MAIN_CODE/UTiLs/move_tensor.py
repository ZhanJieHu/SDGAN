import torch.nn as nn
import torch
def move_all_tensors_to_device(module: nn.Module, device: torch.device):
    module.to(device)
    
    # 显式处理 buffers，防止 register_buffer 遗漏
    for name, buf in module.named_buffers():
        module._buffers[name] = buf.to(device)

    # 处理那些既不是 Parameter 也不是 Buffer 的普通 Tensor 属性
    for name, attr in module.__dict__.items():
        if isinstance(attr, torch.Tensor):
            setattr(module, name, attr.to(device))
            
    # 递归调用子模块
    for child in module.children():
        move_all_tensors_to_device(child, device)


def apply_to_sample(f, sample):
    if hasattr(sample, '__len__') and len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)

# 移动张量的另一个选择。
def move_to_cudapro(sample, device):
    def _move_to_cuda(tensor):
        return tensor.to(device)

    return apply_to_sample(_move_to_cuda, sample)