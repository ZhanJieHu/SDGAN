import os
import shutil

def save_file(file: str, output_path: str, fileName: str):
    """
    保存文件到指定路径
    
    Args:
        file: 文件路径或文件内容
        output_path: 输出目录
        fileName: 文件名
    """
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 构建输出路径
    output_file_path = os.path.join(output_path, fileName)
    
    # 如果file是已存在的文件路径，则复制文件
    if isinstance(file, str) and os.path.isfile(file):
        shutil.copy2(file, output_file_path)
    else:
        # 否则当作内容保存
        mode = 'w' if isinstance(file, str) else 'wb'
        with open(output_file_path, mode) as f:
            f.write(file)
    
    return output_file_path