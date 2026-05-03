from pathlib import Path
import os

def write_txt(file_path, content, mode='w', encoding='utf-8', create_dirs=True):
    """
    写入数据到txt文件
    
    参数:
        file_path: 文件路径（字符串或Path对象）
        content: 要写入的内容（字符串）
        mode: 写入模式，'w'=覆盖写入, 'a'=追加写入 (默认: 'w')
        encoding: 文件编码 (默认: 'utf-8')
        create_dirs: 如果目录不存在是否自动创建 (默认: True)
    
    返回:
        bool: 写入是否成功
    
    示例:
        write_txt('output.txt', 'Hello World')
        write_txt('logs/app.log', 'New log entry', mode='a')
    """
    try:
        # 转换为Path对象
        path = Path(file_path)
        
        # 如果需要，创建父目录
        if create_dirs and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"已创建目录: {path.parent}")
        
        # 写入文件
        with open(path, mode, encoding=encoding) as f:
            f.write(content)
        
        print(f"✓ 成功写入文件: {path}")
        return True
        
    except PermissionError:
        print(f"✗ 错误: 没有权限写入文件 {file_path}")
        return False
    except Exception as e:
        print(f"✗ 错误: {e}")
        return False


def write_txt_lines(file_path, lines, mode='w', encoding='utf-8', create_dirs=True):
    """
    写入多行数据到txt文件（列表形式）
    
    参数:
        file_path: 文件路径
        lines: 要写入的内容列表
        mode: 写入模式 (默认: 'w')
        encoding: 文件编码 (默认: 'utf-8')
        create_dirs: 是否自动创建目录 (默认: True)
    
    返回:
        bool: 写入是否成功
    
    示例:
        lines = ['第一行', '第二行', '第三行']
        write_txt_lines('output.txt', lines)
    """
    try:
        path = Path(file_path)
        
        if create_dirs and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, mode, encoding=encoding) as f:
            for line in lines:
                f.write(str(line) + '\n')
        
        print(f"✓ 成功写入 {len(lines)} 行到文件: {path}")
        return True
        
    except Exception as e:
        print(f"✗ 错误: {e}")
        return False


def append_txt(file_path, content, encoding='utf-8'):
    """
    追加内容到txt文件末尾（快捷函数）
    
    参数:
        file_path: 文件路径
        content: 要追加的内容
        encoding: 文件编码 (默认: 'utf-8')
    
    返回:
        bool: 追加是否成功
    """
    return write_txt(file_path, content, mode='a', encoding=encoding)


# ==================== 使用示例 ====================

def example_basic():
    """基本使用示例"""
    print("=" * 60)
    print("示例1: 基本写入")
    print("=" * 60)
    
    # 简单写入
    write_txt('test_output.txt', 'Hello, World!')
    
    # 写入多行文本
    content = """这是第一行
这是第二行
这是第三行"""
    write_txt('test_multiline.txt', content)


def example_append():
    """追加写入示例"""
    print("\n" + "=" * 60)
    print("示例2: 追加写入")
    print("=" * 60)
    
    # 首次写入
    write_txt('test_append.txt', '第一次写入\n')
    
    # 追加写入
    append_txt('test_append.txt', '第二次追加\n')
    append_txt('test_append.txt', '第三次追加\n')


def example_list():
    """列表写入示例"""
    print("\n" + "=" * 60)
    print("示例3: 写入列表数据")
    print("=" * 60)
    
    # 写入字符串列表
    lines = [
        '姓名: 张三',
        '年龄: 25',
        '城市: 北京'
    ]
    write_txt_lines('test_list.txt', lines)
    
    # 写入数字列表
    numbers = [1, 2, 3, 4, 5]
    write_txt_lines('test_numbers.txt', numbers)


def example_auto_create_dirs():
    """自动创建目录示例"""
    print("\n" + "=" * 60)
    print("示例4: 自动创建目录")
    print("=" * 60)
    
    # 写入到不存在的目录（会自动创建）
    write_txt('output/logs/app.log', 'Application started')
    write_txt('data/2024/report.txt', 'Annual report data')


def example_different_encodings():
    """不同编码示例"""
    print("\n" + "=" * 60)
    print("示例5: 不同编码")
    print("=" * 60)
    
    # UTF-8编码（默认，推荐）
    write_txt('test_utf8.txt', '中文内容测试', encoding='utf-8')
    
    # GBK编码（某些Windows程序需要）
    write_txt('test_gbk.txt', '中文内容测试', encoding='gbk')


def example_real_world():
    """实际应用场景"""
    print("\n" + "=" * 60)
    print("示例6: 实际应用场景")
    print("=" * 60)
    
    # 场景1: 保存日志
    def save_log(message):
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}\n"
        append_txt('app.log', log_entry)
    
    save_log('程序启动')
    save_log('处理数据中...')
    save_log('处理完成')
    
    # 场景2: 保存配置
    config = """# 配置文件
host = localhost
port = 8080
debug = True"""
    write_txt('config.txt', config)
    
    # 场景3: 保存处理结果
    results = [
        '任务1: 成功',
        '任务2: 成功',
        '任务3: 失败'
    ]
    write_txt_lines('results.txt', results)
    
    # 场景4: 保存CSV数据
    csv_data = "姓名,年龄,城市\n张三,25,北京\n李四,30,上海\n"
    write_txt('data.csv', csv_data)


def example_error_handling():
    """错误处理示例"""
    print("\n" + "=" * 60)
    print("示例7: 错误处理")
    print("=" * 60)
    
    # 检查返回值
    success = write_txt('test.txt', 'content')
    if success:
        print("文件写入成功，继续后续操作")
    else:
        print("文件写入失败，执行备用方案")
    
    # 尝试写入到受保护的路径（可能失败）
    write_txt('/root/protected.txt', 'This may fail')


# 运行所有示例
if __name__ == "__main__":
    example_basic()
    example_append()
    example_list()
    example_auto_create_dirs()
    example_different_encodings()
    example_real_world()
    example_error_handling()
    
    print("\n" + "=" * 60)
    print("所有示例执行完成！")
    print("=" * 60)