# 人声提取器 (Vocal Extractor)

这是一个基于频率分析的人声提取应用程序，可以从歌曲中提取人声部分，而不使用传统的反相消除法，效果更自然。做到了类似utagoe的功能，效果更好。

# Open Source Distribution Paused / 暂停开源分发
Thank you for your support and interest in this project.

Starting from the next versions, the source code of this project will **temporarily no longer be publicly distributed**.
This decision was made to allow further development and restructuring of the project.

Please note:

* The **previous open-source versions will remain available** in this repository.
* Future updates may continue to be released as **compiled binaries**.
* The project **may return to open-source in the future**, depending on development plans.

Thank you for your understanding and continued support.

---

感谢大家一直以来对本项目的关注和支持。

从后续版本开始，本项目的**源代码将暂时不再公开分发**。
做出这个决定是为了方便项目后续的开发与结构调整。

需要说明的是：

* **此前已开源的版本仍会保留在仓库中。**
* 未来可能仍会以 **Releases** 形式发布。
* 在合适的情况下，项目 **可能会重新恢复开源**。

感谢大家的理解与支持。

## 功能特点

- 导入歌曲和伴奏文件
- 自动搜索匹配的伴奏文件
- 可调节提取强度
- 实时显示处理进度和状态
- 超快的速度

## 安装要求

- Python 3.7+
- 依赖库：
  - librosa
  - numpy
  - scipy
  - PyQt5
  - soundfile
  - matplotlib

## 使用方法

mac用户直接点击start_app.commad
请预先配置python环境
windows用户在命令行内运行run_app.py
2.0版本多级处理和传统混合算法无任何用处 只是懒得修啦TT

## 工作原理

使用短时傅里叶变换将音频转为频域，计算频谱差值，生成掩码提取人声，再还原到时域。相比传统方法，更自然灵活。代码内有详细注释。

## 注意事项

- 歌曲和伴奏需预处理进行对轨
- 高质量音源效果更好
- 强度值越高，人声越明显，但噪音越大。

## 致谢
[Trae](https://github.com/Trae-AI) 
[Claude](https://github.com/claude) 
