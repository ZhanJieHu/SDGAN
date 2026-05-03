
def _update_cfg_node(cfg_node, **kwargs):
    """
    统一的配置节点更新函数
    替换原有的 .update() 调用
    
    Args:
        cfg_node: YACS CfgNode 配置节点
        **kwargs: 要更新的键值对
        
    Returns:
        更新后的配置节点（原地修改，返回同一个对象）
    """
    # 直接调用 YACS CfgNode 的 update 方法
    # 将关键字参数转换为字典形式
    update_dict = {}
    for key, value in kwargs.items():
        update_dict[key] = value
    
    # 调用原有的 update 方法
    cfg_node.update(update_dict)
    
    return cfg_node

from yacs.config import CfgNode

def _update_cfg_nodePro(cfg_node, **kwargs):
    """
    制造config：_update_cfg_nodePlus(cfg_node=None, ......)
    
    统一的配置节点更新函数
    在函数内部先创建副本，就能避免修改原对象
    
    Args:
        cfg_node: YACS CfgNode 配置节点，可以为 None
        **kwargs: 要更新的键值对
        
    Returns:
        更新后的配置节点（新对象，不影响原对象）
    """
    # 如果 cfg_node 为 None 或空，创建一个新的空 CfgNode
    if cfg_node is None:
        cfg_node = CfgNode()
    else:
        # 关键：创建副本，而不是直接修改原对象
        cfg_node = cfg_node.clone()  # YACS 提供的克隆方法
    
    # 将关键字参数转换为字典形式
    update_dict = {}
    for key, value in kwargs.items():
        update_dict[key] = value
    
    # 调用原有的 update 方法（现在修改的是副本）
    cfg_node.update(update_dict)
    
    return cfg_node

from yacs.config import CfgNode
def _update_cfg_nodePlus(cfg_node, **kwargs):
    """
    制造config：_update_cfg_nodePlus(cfg_node=None, ......)
    
    统一的配置节点更新函数
    替换原有的 .update() 调用
    
    Args:
        cfg_node: YACS CfgNode 配置节点，可以为 None
        **kwargs: 要更新的键值对
        
    Returns:
        更新后的配置节点
    """
    # 如果 cfg_node 为 None 或空，创建一个新的空 CfgNode
    if cfg_node is None:
        cfg_node = CfgNode()
    
    # 将关键字参数转换为字典形式
    update_dict = {}
    for key, value in kwargs.items():
        update_dict[key] = value
    
    # 调用原有的 update 方法
    cfg_node.update(update_dict)
    
    return cfg_node