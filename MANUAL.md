# 基于语义匹配的智能语音寻物系统 (Hajimi AI)

## 1. 项目简介

本项目是一个结合了计算机视觉（Computer Vision）和自然语言处理（NLP）的智能辅助系统。系统能够实时通过摄像头捕捉画面，并利用语音识别技术接收用户的语音指令（例如：“找一下我的手机”）。通过先进的语义匹配算法，系统能在画面中自动定位与语音描述最相似的物体，并进行视觉标注。

该项目旨在解决传统目标检测系统只能识别固定标签的局限性，通过引入语义向量空间，实现了更加自然、模糊的语音交互体验。

## 2. 核心功能

* **实时目标检测**：利用 YOLOv11/v8 模型实时识别视频流中的物体（支持80种常见物体）。
* **离线语音识别**：集成 VOSK 模型，支持无网环境下的中文语音指令识别，保护用户隐私。
* **语义理解与匹配**：使用 Sentence-Transformer 将语音指令和物体标签转化为高维向量，计算余弦相似度，从而实现模糊匹配（例如说“喝水的家伙”能匹配到“杯子”，说“坐的地方”能匹配到“椅子”）。
* **多进程架构**：采用 Python Multiprocessing 技术，将重负载的语音/语义计算与图像渲染分离，保证视频画面的流畅度。

## 3. 环境依赖

本项目基于 Python 3 开发，核心依赖库如下：

* **OpenCV (`opencv-python`)**: 用于视频流捕获、图像处理和绘制。
* **NumPy (`numpy`)**: 用于向量计算和矩阵运算。
* **Pillow (`PIL`)**: 用于在图像上绘制中文字符。
* **Ultralytics YOLO**: 最先进的目标检测模型。
* **VOSK (`vosk`)**: 轻量级离线语音识别引擎。
* **SoundDevice (`sounddevice`)**: 跨平台音频录制库。
* **Sentence-Transformers**: 用于生成文本的语义向量。

## 4. 系统架构与流程图

本系统采用主从进程架构设计，以确保 UI 线程不被阻塞。

```mermaid
graph TD
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style L fill:#bbf,stroke:#333,stroke-width:2px
    style H fill:#bfb,stroke:#333,stroke-width:2px

    A[系统启动 Main Entry] --> B{初始化阶段}
  
    subgraph Child_Process [子进程：语音与语义引擎]
        direction TB
        C1[加载 VOSK 语音模型]
        C2[加载 Sentence-Transformer 模型]
        C3[预计算标签向量库]
        C4[麦克风监听循环]
        C5[语音转文字 (STT)]
        C6[计算语音语义向量]
      
        C1 --> C2 --> C3 --> C4
        C4 --> C5 --> C6
    end
  
    subgraph Main_Process [主进程：视觉与交互核心]
        direction TB
        M1[等待子进程初始化]
        M2[加载 YOLO 视觉模型]
        M3[打开摄像头]
        M4[视频流主循环]
        M5[YOLO 目标检测]
        M6[获取检测结果 (Boxes & Labels)]
        M7[语义相似度匹配 (Dot Product)]
        M8[绘制标注框与 UI]
        M9[显示画面]
      
        M1 --> M2 --> M3 --> M4
        M4 --> M5 --> M6 --> M7 --> M8 --> M9 --> M4
    end

    B -->|启动| Child_Process
    B -->|启动| Main_Process
  
    C3 -- 传输向量库 (Queue) --> M1
    C6 -- 传输指令向量 (Queue) --> M4

```

## 5. 快速开始

### 5.1 安装依赖

在项目根目录下运行以下命令安装所需库：

```bash
pip install -r requirements.txt
```

### 5.2 准备模型文件

请确保项目目录下包含以下模型文件结构：

1. **VOSK 模型**：下载 `vosk-model-small-cn-0.22` 并解压至项目根目录。
2. **YOLO 模型**：确保 `yolo11s.pt` 存在于项目根目录（首次运行会自动下载）。
3. **Embedding 模型**：首次运行时，`sentence-transformers` 会自动下载 `paraphrase-multilingual-MiniLM-L12-v2` 模型。

### 5.3 运行项目

```bash
python main.py
```

运行后，请允许程序访问摄像头和麦克风权限。

## 6. 代码结构说明

| 文件名               | 说明                                                                                                                                         |
| :------------------- | :------------------------------------------------------------------------------------------------------------------------------------------- |
| `main.py`          | **核心代码**。包含 `voice_process_run` (子进程逻辑) 和 `main` (主进程逻辑)。实现了多进程通信、模型加载、实时检测与绘制的所有功能。 |
| `resize_image.py`  | **辅助工具**。用于将原始图片素材调整为适合覆盖在视频上的尺寸。                                                                         |
| `requirements.txt` | **依赖列表**。定义了项目运行所需的 Python 包。                                                                                         |
| `hajimi.png`       | **UI素材**。用于指示目标物体的视觉图标。                                                                                               |

## 7. 关键算法逻辑

### 7.1 语义匹配

系统不进行简单的关键字匹配（String Matching），而是进行语义匹配（Semantic Matching）。

1. 预先计算 YOLO 支持的 80 种物体标签（如 "cup", "chair"）对应的中文语义向量 $V_{label}$。
2. 实时计算用户语音指令（如 "喝水的杯子"）的语义向量 $V_{voice}$。
3. 计算余弦相似度（Cosine Similarity）：
   $$
   Score = \frac{V_{label} \cdot V_{voice}}{\|V_{label}\| \|V_{voice}\|}
   $$
4. 选取 Score 最高且超过阈值（0.3）的物体作为目标。

---

*Hajimi AI Graduation Project Manual*
