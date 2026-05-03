# NUM_CLIPS  = 128
# pooling_list  = [15, 8, 8, 8]
# 变长为NUM_CLIPS的对角线序号为[0, NUM_CLIPS-1]
# pooling_list[i]能覆盖的对角线为：pooling_list[i] * 2^i
# 例如pooling_list[1] = 8， 覆盖对角线条数为8 * 2^1
# 函数功能：保证∑(pooling_list[i] * 2^i) >= NUM_CLIPS-1




def validate_pooling_list(pooling_list, num_clips):
    """
    验证pooling_list是否满足覆盖条件
    
    参数:
        pooling_list: pooling列表
        num_clips: 视频片段数量
    
    功能:
        确保 ∑(pooling_list[i] * 2^i) >= num_clips - 1
        不满足条件则抛出AssertionError
    """
    # 计算当前覆盖的对角线总数
    coverage = sum(pooling_list[i] * (2 ** i) for i in range(len(pooling_list)))
    
    # 目标对角线数
    target = num_clips - 1
    
    # 断言检查
    assert coverage >= target, (
        f"pooling_list覆盖不足！"
        f"当前覆盖: {coverage}, 需要覆盖: {target}, "
        f"差距: {target - coverage}。"
        f"详情: ∑(pooling_list[i] * 2^i) = "
        f"{' + '.join(f'{pooling_list[i]}×2^{i}' for i in range(len(pooling_list)))} = {coverage}"
    )
    
    return True


# 测试示例
if __name__ == "__main__":
    NUM_CLIPS = 128
    
    # 测试1: 满足条件的情况
    pooling_list1 = [15, 8, 8, 8]
    print(f"测试1: NUM_CLIPS={NUM_CLIPS}, pooling_list={pooling_list1}")
    try:
        validate_pooling_list(pooling_list1, NUM_CLIPS)
        coverage = sum(pooling_list1[i] * (2 ** i) for i in range(len(pooling_list1)))
        print(f"✓ 验证通过！覆盖: {coverage} >= {NUM_CLIPS - 1}\n")
    except AssertionError as e:
        print(f"✗ {e}\n")
    
    # 测试2: 不满足条件的情况
    pooling_list2 = [10, 5, 5, 5]
    print(f"测试2: NUM_CLIPS={NUM_CLIPS}, pooling_list={pooling_list2}")
    try:
        validate_pooling_list(pooling_list2, NUM_CLIPS)
        coverage = sum(pooling_list2[i] * (2 ** i) for i in range(len(pooling_list2)))
        print(f"✓ 验证通过！覆盖: {coverage} >= {NUM_CLIPS - 1}\n")
    except AssertionError as e:
        print(f"✗ {e}\n")
    
    # 测试3: 刚好满足条件
    pooling_list3 = [127, 0, 0, 0]
    print(f"测试3: NUM_CLIPS={NUM_CLIPS}, pooling_list={pooling_list3}")
    try:
        validate_pooling_list(pooling_list3, NUM_CLIPS)
        coverage = sum(pooling_list3[i] * (2 ** i) for i in range(len(pooling_list3)))
        print(f"✓ 验证通过！覆盖: {coverage} >= {NUM_CLIPS - 1}\n")
    except AssertionError as e:
        print(f"✗ {e}\n")