#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
人声提取器启动脚本
检查依赖并启动应用程序
"""

import os
import sys
import subprocess
import importlib.util

def check_dependency(module_name):
    """检查依赖是否已安装"""
    return importlib.util.find_spec(module_name) is not None

def main():
    # 检查必要的依赖
    dependencies = ["librosa", "numpy", "scipy", "PyQt5", "soundfile", "matplotlib"]
    missing_deps = [dep for dep in dependencies if not check_dependency(dep)]
    
    if missing_deps:
        print("检测到缺少以下依赖:")
        for dep in missing_deps:
            print(f"  - {dep}")
        
        # 询问用户是否要安装依赖
        response = input("\n是否要安装这些依赖? (y/n): ").strip().lower()
        
        if response == 'y':
            # 获取安装脚本路径
            script_dir = os.path.dirname(os.path.abspath(__file__))
            install_script = os.path.join(script_dir, "install_dependencies.py")
            
            if os.path.exists(install_script):
                print("\n正在安装依赖...")
                try:
                    subprocess.check_call([sys.executable, install_script])
                except subprocess.CalledProcessError:
                    print("\n依赖安装失败，请手动安装依赖后再运行应用程序。")
                    return 1
            else:
                print(f"\n错误: 找不到安装脚本 {install_script}")
                print("请手动安装依赖:")
                print("pip install -r requirements.txt")
                return 1
        else:
            print("\n未安装依赖，应用程序可能无法正常运行。")
    
    # 启动应用程序
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        app_script = os.path.join(script_dir, "vocal_extractor.py")
        
        if not os.path.exists(app_script):
            print(f"\n错误: 找不到应用程序脚本 {app_script}")
            return 1
        
        print("\n正在启动人声提取器...")
        subprocess.check_call([sys.executable, app_script])
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\n启动应用程序时出错: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())