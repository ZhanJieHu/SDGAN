'''
现在这个改进的函数会检测：
参数 (nn.Parameter) - 被 to(device) 移动
缓冲区 (register_buffer) - 被 to(device) 移动
普通张量 (如 self.feat2d_mask2d) - 不被 to(device) 移动
'''
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Set, Tuple

def get_model_all_tensors_device_info(model: nn.Module) -> Tuple[Set[torch.device], Dict[str, torch.device]]:
    """
    分析PyTorch模型中所有张量（参数+缓冲区+普通张量）的设备信息
    """
    all_devices = set()
    tensor_device_map = {}
    
    # 1. 遍历模型的所有参数 (nn.Parameter)
    for name, param in model.named_parameters():
        device = param.device
        all_devices.add(device)
        tensor_device_map[f"参数: {name}"] = device
    
    # 2. 遍历模型的所有缓冲区 (register_buffer)
    for name, buffer in model.named_buffers():
        device = buffer.device
        all_devices.add(device)
        tensor_device_map[f"缓冲区: {name}"] = device
    
    # 3. 遍历模型的所有属性，查找普通张量
    for name, attr in model.__dict__.items():
        if isinstance(attr, torch.Tensor) and not name.startswith('_'):
            # 排除已经是参数或缓冲区的张量
            is_param = any(name in param_name for param_name in model._parameters.keys())
            is_buffer = any(name in buffer_name for buffer_name in model._buffers.keys())
            
            if not is_param and not is_buffer:
                device = attr.device
                all_devices.add(device)
                tensor_device_map[f"普通张量: {name}"] = device
    
    return all_devices, tensor_device_map

def print_model_device_info_detailed(model: nn.Module):
    """
    打印模型的详细设备信息（包括普通张量）
    """
    all_devices, tensor_device_map = get_model_all_tensors_device_info(model)
    
    print("模型详细设备分析结果:")
    print("=" * 60)
    
    # 打印所有出现过的设备
    print("所有出现过的设备:")
    for i, device in enumerate(all_devices, 1):
        print(f"{i}. {device}")
    
    print("\n张量与设备对应关系:")
    print("-" * 50)
    
    # 按设备分组打印张量
    device_tensors = defaultdict(list)
    for tensor_name, device in tensor_device_map.items():
        device_tensors[device].append(tensor_name)
    
    for device, tensors in device_tensors.items():
        print(f"\n设备: {device}")
        print(f"张量数量: {len(tensors)}")
        for tensor_name in tensors:
            print(f"  - {tensor_name}")
    
    # 检查是否所有张量都在同一设备上
    if len(all_devices) == 1:
        print(f"\n✓ 所有张量都在同一设备上: {next(iter(all_devices))}")
    else:
        print(f"\n⚠️ 警告: 张量分布在多个设备上!")
        for device in all_devices:
            count = sum(1 for d in tensor_device_map.values() if d == device)
            print(f"  - {device}: {count}个张量")
