# 2.9修复空字典问题
import torch
import numpy as np
from typing import Any, Dict, List, Tuple, Set, Union, Optional
from abc import ABC, abstractmethod

def else_print(obj, prefix="obj"): # 第一版中用来兜底的打印函数
    print(f"{prefix}: type={type(obj)}", end='')
    try:
        print(f", len={len(obj)}")
    except Exception:
        print()

def all_same_type(lst): # 判断list、tuple中是不是都是一种类型的元素
    # 空列表
    if not lst:  
        return True
    
    # 所有类型相同
    first_type = type(lst[0])
    return all(type(item) == first_type for item in lst) 

def with_kids(input) -> bool:
    '''
    判断input是否有子元素。
    '''
    if (isinstance(input, (dict, list, tuple, set))):
        return True
    else:
        return False
        
class ObjectHandler(ABC):
    """对象处理器的抽象基类"""
    
    def __init__(self, obj: Any, prefix: str, num_toprint: int):
        self.obj = obj
        self.prefix = prefix
        self.num_toprint = num_toprint
    
    @abstractmethod
    def print_current(self) -> None:
        """打印当前对象信息"""
        pass
    
    @abstractmethod
    def go_to_child(self) -> None:
        """处理子对象"""
        pass

class SimpleTypeHandler(ObjectHandler):
    """处理简单类型（int, float, bool, str）"""
    
    def print_current(self) -> None:
        print(f"{self.prefix} = {repr(self.obj)}")
    
    def go_to_child(self) -> None:
        pass  # 简单类型没有子对象

class TensorHandler(ObjectHandler):
    """处理torch.Tensor和np.ndarray"""
    # def print_current(self) -> None: # 原来是按照维度分类处理，现在按照参数量和维度分类处理
    #     print(f"{self.prefix}: type={type(self.obj)}", end='')
    #     try:
    #         if len(self.obj.shape) <= 1:
    #             print(f", value={self.obj.tolist()}")
    #         else:
    #             print(f", shape={self.obj.shape}")
    #     except Exception:
    #         print(f", shape=<unprintable>")


    # max_toprint = 10  # 超过这个参数量就不打印具体数值了
    # def print_current(self) -> None:
    #     print(f"{self.prefix}: type={type(self.obj)}", end='')
    #     try:
    #         param_num = self.obj.numel() if isinstance(self.obj, torch.Tensor) else self.obj.size
    #         if param_num <= self.max_toprint:
    #             print(f", shape={self.obj.shape}, values={self.obj.flatten().tolist()}")
    #         else:
    #             print(f", shape={self.obj.shape}, total_elements={param_num} (too many to display)") 将这里改成打印前几个
    #     except Exception:
    #         print(f", shape=<unprintable>")

    max_toprint = 10  # 超过这个参数量就不打印全部，只打印前几个
    def print_current(self) -> None:
        print(f"{self.prefix}: type={type(self.obj)}", end='')
        try:
            # 获取基本信息
            if isinstance(self.obj, torch.Tensor):
                param_num = self.obj.numel()
                shape = tuple(self.obj.shape)
                vals = self.obj.tolist()
            elif isinstance(self.obj, np.ndarray):
                param_num = self.obj.size
                shape = self.obj.shape
                # vals = self.obj.flatten().tolist()
                vals = self.obj.tolist()
            else:
                param_num = self.obj.size
                shape = getattr(self.obj, "shape", None)
                vals = list(self.obj.flatten()) if hasattr(self.obj, "flatten") else None

            # 按照参数量决定打印内容
            if param_num <= self.max_toprint:
                print(f", shape={shape}, values={vals}")
            else:
                def PrintFirstTensor(t): 
                    """不管 tensor/array 是什么形状，打印第一个数的位置和值
                    兼容 torch.Tensor / numpy.ndarray"""
                    if isinstance(t, torch.Tensor):
                        # 展平后拿第一个
                        first_val = t.flatten()[0].item()
                        first_idx = np.unravel_index(0, tuple(t.shape))
                    elif isinstance(t, np.ndarray):
                        first_val = t.flatten()[0].item()
                        first_idx = np.unravel_index(0, t.shape)
                    else:
                        # 兜底
                        try:
                            flat = list(t.flatten())
                            first_val = flat[0]
                            first_idx = (0,)
                        except Exception:
                            print(", first value=<unprintable>")
                            return
                    print(f", firstElem = {first_val}")

                print(f", shape={shape}", end="")
                PrintFirstTensor(self.obj)

        except Exception:
            print(f", shape=<unprintable>")
        


    def go_to_child(self) -> None:
        pass  # Tensor没有需要递归的子对象

class withShapeHandler(ObjectHandler):
    """处理其他具有shape属性的对象"""
    def print_current(self) -> None:
        # fallback for other shape-based objects
        print(f"{self.prefix}: type={type(self.obj)}", end='')
        shape = getattr(self.obj, 'shape', None)
        try:
            if isinstance(shape, (tuple, list)):
                if len(shape) <= 1:
                    print(f", value={self.obj.tolist()}", end='')
                else:
                    print(f", shape={shape}", end='')
            else:
                print(f", shape={shape}", end='')
        except Exception:
            print(f", shape=<unprintable>", end='')
        print()
    
    def go_to_child(self) -> None:
        pass  # 这种类型没有需要递归的子对象


class SequenceHandler(ObjectHandler):
    """处理序列类型（list和tuple）"""
    
    def print_current(self) -> None:
        # 获取具体的类型名称
        type_name = "list" if isinstance(self.obj, list) else "tuple"
        print(f"{self.prefix}: type={type_name}, len={len(self.obj)}")

    def go_to_child(self) -> None:
        this_sequence = self.obj  # 已经是序列，不需要转换

        # 计算序列的元素个数
        elem_num = len(this_sequence)

        # elems_type记录序列的子元素的类型
        all_simple = all(isinstance(elem, (int, float, bool, str)) for elem in this_sequence)
        has_kids = any(with_kids(elem) for elem in this_sequence)
        
        if all_simple:
            elems_type = "allSimple"
        elif has_kids:
            elems_type = "withKid"
        else:
            elems_type = "else"
        
        def all_same_type(lst):
            # 空列表
            if not lst:  
                return True
            
            # 检查是否所有元素都是Tensor或ndarray
            all_tensors_or_arrays = all(
                isinstance(item, (torch.Tensor, np.ndarray)) for item in lst
            )
            
            if all_tensors_or_arrays:
                # 如果都是Tensor/ndarray，检查形状是否相同
                first_shape = lst[0].shape
                return all(item.shape == first_shape for item in lst)
            else:
                # 不都是Tensor/ndarray。检查每个元素类型是否相同。
                first_type = type(lst[0])
                return all(type(item) == first_type for item in lst)
        
        # 根据 元素类型是否统一、元素个数、元素类型 三个维度判断
        if all_same_type(this_sequence): # 先看元素类型是否统一。若所有元素类型统一说明为普通list、tuple
            if elem_num == 1 and elems_type == "withKid": # 然后同时看元素个数、元素类型
                # 如果只有一个元素, 且这个元素有子元素
                do_pobj(this_sequence[0], prefix=f"{self.prefix}[0]", num_toprint=self.num_toprint, need_print=0)
            elif elem_num < 10 and elems_type == "allSimple":
                # 如果比较少且都是简单类型，直接打印
                print(f"{self.prefix} = {repr(self.obj)}")
            else:
                # 正常处理每个元素
                for i, value in enumerate(this_sequence[:self.num_toprint]):
                    path = f"{self.prefix}[{i}]"
                    do_pobj(value, prefix=path, num_toprint=self.num_toprint, need_print=1)
        else: # 若元素类型不统一例如：[duration, timestamps, audios_name, sentences]，则每个元素都打印
            for i, value in enumerate(this_sequence):
                path = f"{self.prefix}[{i}]"
                do_pobj(value, prefix=path, num_toprint=self.num_toprint, need_print=1)

class DictHandler(ObjectHandler):
    """处理字典类型"""
    
    def print_current(self) -> None:
        print(f"{self.prefix}: type=dict, len={len(self.obj)}")
    
    def go_to_child(self) -> None:
        # 如果字典为空，直接返回
        if len(self.obj) == 0:
            return
    
        # 如果是字典类型，遍历其键值对
        first_key = next(iter(self.obj))  # 获取第一个键
        first_value = self.obj[first_key]  # 获取对应的值

        def normal_print():
            for key, value in self.obj.items():
                path = f"{self.prefix}[{repr(key)}]"
                do_pobj(value, prefix=path, num_toprint=self.num_toprint, need_print=1)

        # 根据元素个数分类：
        if len(self.obj) == 1:# 如果只有一个元素且不是最后一层, 不打印
            if isinstance(first_value, (dict, list, tuple, set)):
                path = f"{self.prefix}[{repr(first_key)}]"

                # 不减少 max_depth 如果只有一个元素
                do_pobj(first_value, prefix=path, num_toprint=self.num_toprint, need_print=0)
            else: # 反之，正常打印
                normal_print()
        elif len(self.obj) < 20: # 如果数量不是特别多。说明是没有大量重复的正常字典。
            normal_print()
            
        else: # 如果数量特别多。说明字典中各个元素是非常相似的，打印一个就行。
            for key, value in self.obj.items():
                path = f"{self.prefix}[{repr(key)}]"
                do_pobj(value, prefix=path, num_toprint=self.num_toprint, need_print=1)
                break

class SetHandler(ObjectHandler): #原来set和tuple的处理一样。现在tuple的处理变成和list一样了。
    """处理集合类型"""
    
    def print_current(self) -> None:
        print(f"{self.prefix}: type=set, len={len(self.obj)}")
    
    def go_to_child(self) -> None:
        iterable = list(self.obj)
        for i, value in enumerate(iterable):
            path = f"{self.prefix}[{i}]"
            do_pobj(value, prefix=path, num_toprint=self.num_toprint, need_print=1)

class ObjectAttrHandler(ObjectHandler):
    """处理普通对象属性"""
    
    def print_current(self) -> None:
        print(f"{self.prefix}: type={type(self.obj)}")
    
    def go_to_child(self) -> None:
        if hasattr(self.obj, '__dict__'):
            for attr in vars(self.obj):
                value = getattr(self.obj, attr)
                path = f"{self.prefix}.{attr}"
                do_pobj(value, prefix=path, num_toprint=self.num_toprint, need_print=1)


def get_handler(obj: Any, prefix: str, num_toprint: int) -> ObjectHandler:
    """根据对象类型获取对应的处理器"""
    if isinstance(obj, (int, float, bool, str)):
        return SimpleTypeHandler(obj, prefix, num_toprint)
    elif isinstance(obj, (torch.Tensor, np.ndarray)):
        return TensorHandler(obj, prefix, num_toprint)
    elif hasattr(obj, "shape"): # fallback for other shape-based objects
        return withShapeHandler(obj, prefix, num_toprint)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return SequenceHandler(obj, prefix, num_toprint)
    elif isinstance(obj, dict):
        return DictHandler(obj, prefix, num_toprint)
    elif isinstance(obj, set):
        return SetHandler(obj, prefix, num_toprint)
    else:
        return ObjectAttrHandler(obj, prefix, num_toprint) 
    # elif hasattr(obj, '__dict__'):  # 放在最后作为兜底
    #     return ObjectAttrHandler(obj, prefix, num_toprint)


def do_pobj(
    obj: Any,
    prefix: str = "obj",
    num_toprint: int = 1,
    need_print: bool = True,
) -> None:
    """
    深度优先打印对象及其子对象的类型和长度。
    """
      
    # 1. 根据类型获取对应的处理器
    handler = get_handler(obj, prefix, num_toprint)
    
    # 2. 打印当前对象信息
    if need_print:
        handler.print_current()
    
    # 3. 处理子对象
    handler.go_to_child()

def pobj(
    obj: Any,
    prefix: str = "obj",
    ptype: str = "#", # 打印类型
    num_toprint: int = 1,
    need_print: bool = True,
) -> None:
    if "#" in ptype:
        prefix = "# " + prefix
    elif "'" in ptype or "‘" in ptype or "’" in ptype: # 包含单引号
        prefix = '' + prefix
        print("'''")
    elif '"' in ptype or "“" in ptype or "”" in ptype: # 包含双引号
        prefix = '' + prefix
        print('"""')
    
    # 深度优先打印对象及其子对象的类型和长度。
    do_pobj(obj=obj, prefix=prefix, num_toprint=num_toprint, need_print=need_print)
    
    if "'" in ptype or "‘" in ptype or "’" in ptype: # 包含单引号
        print("'''")
    elif '"' in ptype or "“" in ptype or "”" in ptype: # 包含双引号
        print('"""')


# -------------------- 使用示例 --------------------
if __name__ == "__main__":
    # 1. 普通字典（字符串键）
    person = {"name": "张三", "age": 25, "gender": "男"}
    print(person)  # 输出：{'name': '张三', 'age': 25, 'gender': '男'}

    # 2. 数字键（键唯一，重复键会覆盖前值）
    score = {1: 90, 2: 85, 3: 95, 1: 100}  # 键 1 重复，最终保留最后一个值
    print(score)  # 输出：{1: 100, 2: 85, 3: 95}

    # 3. 元组键（元组不可变，可作为键）
    location = {("北京", 116.4): "东八区", ("纽约", -74.0): "西五区"}
    print(location)  # 输出：{('北京', 116.4): '东八区', ('纽约', -74.0): '西五区'}

    # 4. 空字典
    empty_dict = {}
    print(type(empty_dict))  # 输出：<class 'dict'>（确认是字典类型）

    dicts = (empty_dict, person, score, location, empty_dict)

    pobj(dicts)

