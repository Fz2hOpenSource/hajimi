# 哈基米之萝卜-纸巾难题 

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)

哈基米之萝卜-纸巾 AI 是一个有趣的互动式 AI 助手，它结合了**计算机视觉**和**语音识别**技术。当你对着麦克风说出物品名称时（例如“杯子”、“猫”），屏幕上的“哈基米”（一只可爱的虚拟猫）会自动跳到该物品旁边并指向它。

本项目展示了多模态交互的简单实现：将语音指令转化为语义向量，与画面中物体的标签向量进行匹配，从而实现“听懂”并“看见”你的指令。

## ✨ 功能特点

*   **实时目标检测**：使用 YOLOv11 快速识别视频流中的物体。
*   **离线语音识别**：集成 Vosk 模型，支持无需联网的中文语音指令识别。
*   **语义理解**：使用 SentenceTransformers 进行文本向量化，支持模糊匹配（例如说“那个喝水的家伙”可能匹配到“杯子”）。
*   **现代化 UI**：基于 CustomTkinter 构建的深色模式界面，实时显示识别日志。
*   **可爱交互**：内置虚拟角色“哈基米”与现实物体互动。

## 🛠️ 安装指南

### 1. 克隆仓库

```bash
git clone https://github.com/your-username/hajimi-ai.git
cd hajimi-ai
```

### 2. 安装依赖

请确保安装了 Python 3.8 或更高版本。

```bash
pip install opencv-python numpy sounddevice vosk sentence-transformers ultralytics customtkinter pillow
```

### 3. 下载模型文件

本项目依赖以下模型文件，请下载并放置在项目根目录或指定路径下：

*   **YOLO 模型**: 下载 `yolo11s.pt` (或者其他版本) 并放在项目根目录。
    *   来源: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
*   **Vosk 语音模型**: 下载中文模型 (例如 `vosk-model-small-cn-0.22`) 并解压到项目根目录。
    *   下载地址: [Vosk Models](https://alphacephei.com/vosk/models)
    *   *注意：解压后的文件夹名称需与代码配置一致，默认为 `vosk-model-small-cn-0.22`*

## 🚀 运行方法

确保摄像头和麦克风已连接，然后运行：

```bash
python hajimi.py
```

*   **左侧面板**：显示系统日志和语音识别状态。
*   **右侧画面**：实时视频预览。
*   **交互方式**：对着麦克风说出画面中存在的物体（如“手机”、“鼠标”），哈基米会指向识别到的物体。

## 🧩 技术栈与致谢

本项目使用了以下优秀的开源项目和模型：

### 计算机视觉 (Computer Vision)
*   **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)**:用于实时物体检测。
    *   *License: AGPL-3.0*

### 语音识别 (Speech Recognition)
*   **[Vosk API](https://alphacephei.com/vosk/)**: 用于离线语音转文本。
    *   *License: Apache 2.0*

### 自然语言处理 (NLP)
*   **[SentenceTransformers](https://www.sbert.net/)**: 用于生成文本嵌入向量，实现语义匹配。
    *   *Model: paraphrase-multilingual-MiniLM-L12-v2*
    *   *License: Apache 2.0*

### 图形界面 (GUI)
*   **[CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)**: 用于构建现代化的 Python UI。
    *   *License: MIT*

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源。

---
*Made with ❤️ by Fz2hOpensource Team*
