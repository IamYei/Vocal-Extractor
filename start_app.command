#!/bin/bash

# 人声提取器启动脚本
# 在macOS上可以直接双击运行

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 切换到脚本目录
cd "$SCRIPT_DIR"

# 显示欢迎信息
echo "=================================================="
echo "            人声提取器启动程序                  "
echo "=================================================="
echo ""

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python 3。请安装Python 3后再运行此程序。"
    echo "您可以从 https://www.python.org/downloads/ 下载Python。"
    echo ""
    echo "按任意键退出..."
    read -n 1
    exit 1
fi

# 运行启动脚本
echo "正在启动应用程序..."
python3 "$SCRIPT_DIR/run_app.py"

# 如果应用程序异常退出，等待用户按键
if [ $? -ne 0 ]; then
    echo ""
    echo "应用程序异常退出。请查看上面的错误信息。"
    echo "按任意键关闭此窗口..."
    read -n 1
fi