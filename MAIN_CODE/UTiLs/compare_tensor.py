import torch
import numpy as np

def compare_tensors(tensor1, tensor2, name1="Tensor1", name2="Tensor2", rtol=1e-5, atol=1e-8):
    """
    对比两个PyTorch tensor是否完全一样（形状、值、梯度等）
    
    参数:
        tensor1: 第一个PyTorch tensor
        tensor2: 第二个PyTorch tensor
        name1, name2: tensor的名称，用于输出
        rtol: 相对容差
        atol: 绝对容差
    """
    print(f"\n{'='*60}")
    print(f"对比 {name1} 和 {name2}")
    print(f"{'='*60}")
    
    all_match = True
    
    # 1. 检查类型
    type1 = type(tensor1).__name__
    type2 = type(tensor2).__name__
    print(f"类型: {name1}={type1}, {name2}={type2}")
    
    if not isinstance(tensor1, torch.Tensor) or not isinstance(tensor2, torch.Tensor):
        print("❌ 输入不是PyTorch tensor!")
        return False
    
    # 2. 检查设备
    device1 = tensor1.device
    device2 = tensor2.device
    print(f"设备: {name1}={device1}, {name2}={device2}")
    if device1 != device2:
        print("❌ 设备不匹配!")
        all_match = False
    
    # 3. 检查数据类型
    dtype1 = tensor1.dtype
    dtype2 = tensor2.dtype
    print(f"数据类型: {name1}={dtype1}, {name2}={dtype2}")
    if dtype1 != dtype2:
        print("❌ 数据类型不匹配!")
        all_match = False
    
    # 4. 检查形状
    shape1 = tensor1.shape
    shape2 = tensor2.shape
    print(f"形状: {name1}={shape1}, {name2}={shape2}")
    
    if shape1 != shape2:
        print("❌ 形状不匹配!")
        all_match = False
        return all_match
    else:
        print("✅ 形状匹配")
    
    # 5. 检查值是否相等
    print("数值比较:")
    
    # 移动到CPU并转换为numpy
    np1 = tensor1.detach().cpu().numpy()
    np2 = tensor2.detach().cpu().numpy()
    
    # 使用numpy的allclose进行比较
    values_close = np.allclose(np1, np2, rtol=rtol, atol=atol)
    max_diff = np.max(np.abs(np1 - np2))
    mean_diff = np.mean(np.abs(np1 - np2))
    
    print(f"  - 最大差异: {max_diff:.2e}")
    print(f"  - 平均差异: {mean_diff:.2e}")
    print(f"  - 容差检查 (rtol={rtol}, atol={atol}): {'✅ 通过' if values_close else '❌ 失败'}")
    
    if not values_close:
        all_match = False
        # 找出差异最大的位置
        diff_matrix = np.abs(np1 - np2)
        max_idx = np.unravel_index(np.argmax(diff_matrix), diff_matrix.shape)
        print(f"  - 最大差异位置: {max_idx}")
        print(f"  - 该位置值: {name1}[{max_idx}] = {np1[max_idx]}, {name2}[{max_idx}] = {np2[max_idx]}")
    else:
        print("✅ 数值匹配")
    
    # 6. 检查梯度
    print("梯度检查:")
    
    grad1 = tensor1.grad
    grad2 = tensor2.grad
    
    if grad1 is None and grad2 is None:
        print("  ✅ 两个tensor都没有梯度")
    elif grad1 is None and grad2 is not None:
        print(f"  ❌ {name1}梯度为None, {name2}有梯度")
        all_match = False
    elif grad1 is not None and grad2 is None:
        print(f"  ❌ {name1}有梯度, {name2}梯度为None")
        all_match = False
    else:
        # 都有梯度，比较梯度
        grad_np1 = grad1.detach().cpu().numpy()
        grad_np2 = grad2.detach().cpu().numpy()
        
        grad_close = np.allclose(grad_np1, grad_np2, rtol=rtol, atol=atol)
        grad_max_diff = np.max(np.abs(grad_np1 - grad_np2))
        
        print(f"  - 梯度最大差异: {grad_max_diff:.2e}")
        print(f"  - 梯度匹配: {'✅ 是' if grad_close else '❌ 否'}")
        
        if not grad_close:
            all_match = False
        else:
            print("✅ 梯度匹配")
    
    # 7. 检查requires_grad
    req_grad1 = tensor1.requires_grad
    req_grad2 = tensor2.requires_grad
    print(f"requires_grad: {name1}={req_grad1}, {name2}={req_grad2}")
    if req_grad1 != req_grad2:
        print("❌ requires_grad不匹配!")
        all_match = False
    else:
        print("✅ requires_grad匹配")
    
    # 总结
    print(f"\n{'='*60}")
    if all_match:
        print(f"🎉 所有检查项都匹配! {name1} 和 {name2} 完全一样")
    else:
        print(f"⚠️  存在不匹配的项! {name1} 和 {name2} 不完全一样")
    print(f"{'='*60}")
    
    return all_match

# 测试示例
if __name__ == "__main__":
    print("测试对比函数:")
    
    # 测试1: 完全相同的tensor
    print("\n1. 测试完全相同的tensor:")
    torch_tensor1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    torch_tensor2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    
    # 添加一些梯度用于测试
    torch_tensor1.grad = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    torch_tensor2.grad = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    
    compare_tensors(torch_tensor1, torch_tensor2, "Tensor1", "Tensor2")
    
    # 测试2: 数值不同的tensor
    print("\n2. 测试数值不同的tensor:")
    torch_tensor3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    torch_tensor4 = torch.tensor([[1.0, 2.1], [3.0, 4.0]])  # 第二个元素不同
    compare_tensors(torch_tensor3, torch_tensor4, "Tensor3", "Tensor4")
    
    # 测试3: 形状不同的tensor
    print("\n3. 测试形状不同的tensor:")
    torch_tensor5 = torch.tensor([[1.0, 2.0]])
    torch_tensor6 = torch.tensor([[1.0], [2.0]])
    compare_tensors(torch_tensor5, torch_tensor6, "Tensor5", "Tensor6")
    
    # 测试4: 梯度不同的tensor
    print("\n4. 测试梯度不同的tensor:")
    torch_tensor7 = torch.tensor([[1.0, 2.0]], requires_grad=True)
    torch_tensor8 = torch.tensor([[1.0, 2.0]], requires_grad=True)
    torch_tensor7.grad = torch.tensor([[0.1, 0.2]])
    torch_tensor8.grad = torch.tensor([[0.1, 0.3]])  # 梯度不同
    compare_tensors(torch_tensor7, torch_tensor8, "Tensor7", "Tensor8")
    
    # 测试5: requires_grad不同的tensor
    print("\n5. 测试requires_grad不同的tensor:")
    torch_tensor9 = torch.tensor([[1.0, 2.0]], requires_grad=True)
    torch_tensor10 = torch.tensor([[1.0, 2.0]], requires_grad=False)
    compare_tensors(torch_tensor9, torch_tensor10, "Tensor9", "Tensor10")