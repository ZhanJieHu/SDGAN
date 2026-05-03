import torch

def check_tensor_memory(tensor1):
    print(f"id(tensor1): {id(tensor1)}")
    print(f"tensor1.data_ptr(): {tensor1.data_ptr()}")
    print(f"torch.equal(tensor1): {torch.equal(tensor1)}")
    print(f"tensor1.shape: {tensor1.shape}")
    print(f"tensor1.is_contiguous(): {tensor1.is_contiguous()}")
    print(f"tensor1.stride(): {tensor1.stride()}")
    '''
    stride 告诉你：要从当前元素「跳到同一维度下一个元素」，在内存里需要跨过多少个实际的存储位置。
    或者说：每个维度上，步长是多少
    print("stride:", x.stride())   # (4, 1)
    第 0 维（行）移动一步（从第 0 行 → 第 1 行），要在内存里跳过 4 个元素（因为每行有 4 个数）
    第 1 维（列）移动一步（从左往右一个格子），在内存里只跳过 1 个元素（连续存储的）
    '''

def check_twoTensor(tensor1, tensor2, name1="原tensor", name2="新tensor"):
    print(f"\n→ {name1}  vs  {name2}:")
    print(f"   相同对象      : {id(tensor1) == id(tensor2)}")
    print(f"   共享内存      : {tensor1.data_ptr() == tensor2.data_ptr()}")
    print(f"   内容相等      : {torch.equal(tensor1, tensor2)}")
    print(f"   shape         : {tensor1.shape}  →  {tensor2.shape}")
    print(f"   is_contiguous : {tensor1.is_contiguous()} → {tensor2.is_contiguous()}")
    print(f"   stride        : {tensor1.stride()} → {tensor2.stride()}")
    '''
    stride 告诉你：要从当前元素「跳到同一维度下一个元素」，在内存里需要跨过多少个实际的存储位置。
    或者说：每个维度上，步长是多少
    print("stride:", x.stride())   # (4, 1)
    第 0 维（行）移动一步（从第 0 行 → 第 1 行），要在内存里跳过 4 个元素（因为每行有 4 个数）
    第 1 维（列）移动一步（从左往右一个格子），在内存里只跳过 1 个元素（连续存储的）
    '''

if __name__ == "__main__":
    x = torch.arange(12).reshape(3, 4).float()
    # 非连续 tensor 上 reshape 也会自动变成 contiguous（复制）
    xt = x.transpose(0, 1)      # 非连续
    print("\n先 transpose 得到非连续 tensor:")
    check_twoTensor(x, xt, "x", "x.transpose(0,1)")
    xr2 = xt.reshape(-1)        # 变成一维向量
    check_twoTensor(xt, xr2, "xt(non-contiguous)", "xt.reshape(-1) → 自动复制")