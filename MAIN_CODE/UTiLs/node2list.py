def nclip2poolinglist(node: int) -> list:
    """
    将2的n次幂的node转换为列表
    
    参数:
        node: 2的n次幂，最小为16
    
    返回:
        列表，第一个元素为15，之后根据node大小添加8
    
    示例:
        node2list(16) -> [15]
        node2list(32) -> [15, 8]
        node2list(64) -> [15, 8, 8]
        node2list(128) -> [15, 8, 8, 8]
    """
    # 检查node是否为2的幂且不小于16
    if node < 2 or (node & (node - 1)) != 0:
        raise ValueError("node必须是2的n次幂")
    
    if node < 16:
        return [node - 1] # too few to sparse
    
    # 计算需要多少个8
    # node=16(2^4)时，有0个8
    # node=32(2^5)时，有1个8
    # node=64(2^6)时，有2个8
    # node=128(2^7)时，有3个8
    # 规律：8的个数 = 幂次 - 4
    import math
    power = int(math.log2(node))
    num_eights = power - 4
    
    # 构建结果列表
    result = [15] + [8] * num_eights
    
    return result


# 测试代码
if __name__ == "__main__":
    test_cases = [16, 32, 64, 128, 256, 512]
    
    for node in test_cases:
        result = nclip2poolinglist(node)
        print(f"node2list({node}) = {result}")
        
        # 验证公式：15 + 求和(8*2^n) = node - 1
        # 其中第i个8乘以2^(i+1)，i从0开始
        total = 15
        for i, val in enumerate(result[1:]):  # 跳过第一个15
            total += val * (2 ** (i + 1))
        
        expected = node - 1
        status = "✓" if total == expected else "✗"
        print(f"  验证: 15 + sum(8*2^n) = {total}, 期望 {expected} {status}")
        print()