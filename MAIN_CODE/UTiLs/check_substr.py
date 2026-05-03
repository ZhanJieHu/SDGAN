def check_substring(main_str: str, sub_str: str, case_sensitive=False) -> bool:
    """
    检查主字符串中是否包含连续的子字符串，可控制是否区分大小写
    
    参数:
        main_str: 待检查的主字符串
        sub_str: 要查找的连续子字符串
        case_sensitive: 布尔值，True表示区分大小写，False表示不区分
    
    返回:
        bool: 存在返回True，不存在返回False
    """
    # 处理空字符串情况：如果子字符串为空，直接返回True
    if not sub_str:
        return True
    
    # 根据case_sensitive决定是否转换大小写
    if not case_sensitive:  # 不区分大小写时，统一转小写
        main_str_lower = main_str.lower()
        sub_str_lower = sub_str.lower()
        return sub_str_lower in main_str_lower
    else:  # 区分大小写时，直接判断
        return sub_str in main_str

# 测试用例
if __name__ == "__main__":
    # 测试1：区分大小写，存在匹配
    print(check_substring("activitynet", "activitynet", True))  # 输出: True
    # 测试2：区分大小写，不存在匹配
    print(check_substring("activitynet", "aNet", True))  # 输出: False
    # 测试3：不区分大小写，存在匹配
    print(check_substring("S_aNet_vr", "aNet", False)) # 输出: True
    print(check_substring("S_aNet_vr", "anet", False))  # 输出: True
    print(check_substring("S_aNet", "anet", False))  # 输出: True
    print(check_substring("S_aNet_vr", "vr", False))  # 输出: True
    # 测试4：子字符串为空
    print(check_substring("Hello", "", True))             # 输出: True
    # 测试5：完全不匹配
    print(check_substring("activity_net", "activitynet", True))        # 输出: False