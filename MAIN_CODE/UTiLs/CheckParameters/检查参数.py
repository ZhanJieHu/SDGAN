'''
get_model_parameters_device_info 函数只遍历了 model.named_parameters()，
PyTorch 中：
named_parameters(): 只返回 nn.Parameter 对象（需要梯度的参数）
个普通的 torch.Tensor，不是 nn.Parameter
普通张量不会被 model.to(device) 自动移动



'''
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Set, Tuple

def get_model_parameters_device_info(model: nn.Module) -> Tuple[Set[torch.device], Dict[str, torch.device]]:
    """
    分析PyTorch模型中所有参数的设备信息
    
    参数:
        model: PyTorch模型
    
    返回:
        Tuple[Set[torch.device], Dict[str, torch.device]]: 
            - 所有出现过的设备集合
            - 参数名称到设备的映射字典
    """
    # 存储所有出现过的设备
    all_devices = set()
    # 存储参数名称到设备的映射
    param_device_map = {}
    
    # 遍历模型的所有参数
    for name, param in model.named_parameters():
        # 获取参数的设备
        device = param.device
        # 添加到设备集合
        all_devices.add(device)
        # 添加到映射字典
        param_device_map[name] = device
    
    return all_devices, param_device_map

import torch.nn as nn
def print_model_device_info(model: nn.Module):
    import torch
    import torch.nn as nn
    from collections import defaultdict
    from typing import Dict, List, Set, Tuple

    """
    打印模型的设备信息
    
    参数:
        model: PyTorch模型
    """
    all_devices, param_device_map = get_model_parameters_device_info(model)
    
    print("模型参数设备分析结果:")
    print("=" * 50)
    
    # 打印所有出现过的设备
    print("所有出现过的设备:")
    for i, device in enumerate(all_devices, 1):
        print(f"{i}. {device}")
    
    print("\n参数与设备对应关系:")
    print("-" * 40)
    
    # 按设备分组打印参数
    device_params = defaultdict(list)
    for param_name, device in param_device_map.items():
        device_params[device].append(param_name)
    
    for device, params in device_params.items():
        print(f"\n设备: {device}")
        print(f"参数数量: {len(params)}")
        for param_name in params:
            print(f"  - {param_name}")
    
    # 检查是否所有参数都在同一设备上
    if len(all_devices) == 1:
        print(f"\n✓ 所有参数都在同一设备上: {next(iter(all_devices))}")
    else:
        print(f"\n⚠️ 警告: 参数分布在多个设备上!")
        for device in all_devices:
            count = sum(1 for d in param_device_map.values() if d == device)
            print(f"  - {device}: {count}个参数")

# 示例用法
if __name__ == "__main__":
    # 创建一个示例模型
    class SampleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)
        
        def forward(self, x):
            return self.fc2(self.fc1(x))
    
    # 创建模型实例
    model = SampleModel()
    
    # 将部分参数移动到不同设备（如果有GPU的话）
    if torch.cuda.is_available():
        model.fc1 = model.fc1.to('cuda:0')
        model.fc2 = model.fc2.to('cuda:0' if torch.cuda.device_count() == 1 else 'cuda:1')
    
    # 分析并打印设备信息
    print_model_device_info(model)
    
    # 也可以直接获取设备信息
    all_devices, param_device_map = get_model_parameters_device_info(model)
    print(f"\n直接获取的设备信息:")
    print(f"所有设备: {all_devices}")
    print(f"参数设备映射: {param_device_map}")