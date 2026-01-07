import cv2
import numpy as np
import multiprocessing as mp
import sys
import os
import time
import json
import platform
import webbrowser
import customtkinter as ctk
from PIL import Image, ImageDraw, ImageFont, ImageTk
from ultralytics import YOLO

# ================= 配置区域 =================
# 获取当前脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VOSK_MODEL_PATH = os.path.join(BASE_DIR, "vosk-model-small-cn-0.22")
# 尝试查找当前目录下的 yolo11*.pt 文件，默认使用 s 版本
YOLO_MODEL_NAME = os.path.join(BASE_DIR, "yolo11n.pt")
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# 个人主页链接
USER_HOMEPAGE = "https://github.com/your-username" 

def get_font_path():
    """根据操作系统自动选择中文字体路径"""
    system = platform.system()
    if system == "Windows":
        return "C:/Windows/Fonts/msyh.ttc" # 微软雅黑
    elif system == "Darwin": # macOS
        return "/System/Library/Fonts/PingFang.ttc" # 苹方
    elif system == "Linux":
        # 尝试一些常见的 Linux 中文字体路径
        paths = [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
        ]
        for p in paths:
            if os.path.exists(p):
                return p
    return None # 此时将使用 PIL 默认字体

FONT_PATH = get_font_path()

def check_models():
    """检查必要的模型文件是否存在，不存在则打印下载链接"""
    missing = []
    global YOLO_MODEL_NAME
    # 1. 检查 YOLO 模型
    # 如果指定路径不存在，尝试搜寻同目录下的其他 pt 文件
    if not os.path.exists(YOLO_MODEL_NAME):
        found = False
        for f in os.listdir(BASE_DIR):
            if f.endswith(".pt") and "yolo" in f.lower():

                YOLO_MODEL_NAME = os.path.join(BASE_DIR, f)
                found = True
                print(f"[Info] 未找到 yolo11s.pt，自动使用: {f}")
                break
        if not found:
            missing.append({
                "name": "YOLOv11 模型 (yolo11s.pt)",
                "url": "https://github.com/ultralytics/ultralytics",
                "path": "项目根目录"
            })

    # 2. 检查 Vosk 模型
    # 检查 VOSK_MODEL_PATH 是否存在，且里面有 conf 文件夹
    # 也要兼容解压后多套一层的情况
    global VOSK_MODEL_PATH
    valid_vosk = False
    if os.path.exists(VOSK_MODEL_PATH):
        if os.path.exists(os.path.join(VOSK_MODEL_PATH, "conf")):
            valid_vosk = True
        elif os.path.exists(os.path.join(VOSK_MODEL_PATH, os.path.basename(VOSK_MODEL_PATH), "conf")):
            # 修正路径
            VOSK_MODEL_PATH = os.path.join(VOSK_MODEL_PATH, os.path.basename(VOSK_MODEL_PATH))
            valid_vosk = True
            
    if not valid_vosk:
         missing.append({
                "name": "Vosk 中文语音模型 (vosk-model-small-cn-0.22)",
                "url": "https://alphacephei.com/vosk/models",
                "path": "项目根目录 (解压后)"
            })
            
    if missing:
        print("\n" + "="*50)
        print("❌ 错误：缺少必要的模型文件，程序无法启动。")
        print("请下载以下文件并放置在正确位置：")
        print("="*50)
        for item in missing:
            print(f"\n📦 {item['name']}")
            print(f"   🔗 下载地址: {item['url']}")
            print(f"   📂 放置位置: {item['path']}")
        print("\n" + "="*50)
        print(f"💡 更多信息请访问作者主页: {USER_HOMEPAGE}")
        input("\n按回车键退出...")
        sys.exit(1)

# 在导入模块后立即检查
check_models()

# 英文 -> 中文 缓存字典
EN_ZH_CACHE = {
    "person": "人", "bicycle": "自行车", "car": "汽车", "motorcycle": "摩托车", "airplane": "飞机",
    "bus": "公交车", "train": "火车", "truck": "卡车", "boat": "船", "traffic light": "红绿灯",
    "fire hydrant": "消防栓", "stop sign": "停车标志", "parking meter": "停车计时器", "bench": "长椅",
    "bird": "鸟", "cat": "猫", "dog": "狗", "horse": "马", "sheep": "羊", "cow": "牛",
    "elephant": "大象", "bear": "熊", "zebra": "斑马", "giraffe": "长颈鹿", "backpack": "背包",
    "umbrella": "雨伞", "handbag": "手提包", "tie": "领带", "suitcase": "手提箱", "frisbee": "飞盘",
    "skis": "滑雪板", "snowboard": "单板滑雪", "sports ball": "球", "kite": "风筝",
    "baseball bat": "棒球棒", "baseball glove": "棒球手套", "skateboard": "滑板",
    "surfboard": "冲浪板", "tennis racket": "网球拍", "bottle": "瓶子", "wine glass": "酒杯",
    "cup": "杯子", "fork": "叉子", "knife": "刀", "spoon": "勺子", "bowl": "碗", "banana": "香蕉",
    "apple": "苹果", "sandwich": "三明治", "orange": "橘子", "broccoli": "西兰花", "carrot": "胡萝卜",
    "hot dog": "热狗", "pizza": "披萨", "donut": "甜甜圈", "cake": "蛋糕", "chair": "椅子",
    "couch": "沙发", "potted plant": "盆栽", "bed": "床", "dining table": "餐桌", "toilet": "马桶",
    "tv": "电视", "laptop": "笔记本电脑", "mouse": "鼠标", "remote": "遥控器", "keyboard": "键盘",
    "cell phone": "手机", "microwave": "微波炉", "oven": "烤箱", "toaster": "烤面包机", "sink": "水槽",
    "refrigerator": "雪柜", "book": "书", "clock": "时钟", "vase": "花瓶", "scissors": "剪刀",
    "teddy bear": "泰迪熊", "hair drier": "吹风机", "toothbrush": "牙刷"
}

# ================= 子进程：语音与语义处理 =================
class QueueLogger:
    """重定向 stdout/stderr 到主进程的队列，防止 Windows 句柄无效错误"""
    def __init__(self, queue, prefix="[Child]"):
        self.queue = queue
        self.prefix = prefix
    
    def write(self, message):
        if message.strip():
            self.queue.put(("log", f"{self.prefix} {message.strip()}"))
            
    def flush(self):
        pass

def voice_process_run(msg_queue, cache_items, vosk_path, embed_model_name):
    """
    运行在独立进程中：
    1. 加载 Vosk 和 SentenceTransformer
    2. 计算缓存字典的向量并发送给主进程
    3. 监听麦克风 -> 转文字 -> 转向量 -> 发送给主进程
    """
    # 重定向标准输出，防止 [WinError 6] 句柄无效
    sys.stdout = QueueLogger(msg_queue, "[Child]")
    sys.stderr = QueueLogger(msg_queue, "[Child Error]")
    
    print("正在初始化语音与语义引擎...")
    
    try:
        from sentence_transformers import SentenceTransformer
        import sounddevice as sd
        import vosk
        
        # 静音 Vosk 底层日志
        vosk.SetLogLevel(-1)
    except ImportError as e:
        msg_queue.put(("error", f"缺少依赖: {e}"))
        return

    # 1. 加载模型
    try:
        # 加载向量模型
        embedder = SentenceTransformer(embed_model_name)
        print("向量模型加载完成")

        # 加载语音模型
        final_path = vosk_path
        # 自动检测嵌套目录 (例如解压时多了一层 vosk-model-small-cn-0.22)
        if os.path.exists(os.path.join(vosk_path, vosk_path)):
             final_path = os.path.join(vosk_path, vosk_path)
        
        if not os.path.exists(final_path) or not os.path.exists(os.path.join(final_path, "conf")):
            msg_queue.put(("error", f"无效的 Vosk 模型路径: {final_path} (请检查 conf 文件夹)"))
            return
            
        vosk_model = vosk.Model(final_path)
        print(f"Vosk 语音模型加载完成 (路径: {final_path})", flush=True)

    except Exception as e:
        msg_queue.put(("error", f"模型加载失败: {e}"))
        return

    # 2. 预计算所有类别的中文向量
    # 提取所有唯一的中文标签
    unique_labels = list(set(cache_items.values()))
    print(f"正在预计算 {len(unique_labels)} 个类别的向量...")
    
    label_vectors = {}
    for label in unique_labels:
        vec = embedder.encode(label, normalize_embeddings=True)
        label_vectors[label] = vec
    
    # 发送初始化数据回主进程
    msg_queue.put(("init_vectors", label_vectors))
    print("初始化向量已发送")

    # 3. 开启音频监听循环
    q_audio = mp.Queue()

    def audio_callback(indata, frames, time, status):
        if status:
            # 这里的 print 也会被重定向
            print(f"Audio Error: {status}")
        q_audio.put(bytes(indata))

    try:
        samplerate = 16000
        rec = vosk.KaldiRecognizer(vosk_model, samplerate)
        
        print("🎤 麦克风监听中...")
        devices = sd.query_devices()
        print(devices)  
        # 使用 sounddevice 开启流
        with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=2,
                               dtype='int16', channels=1, callback=audio_callback):
            while True:
                data = q_audio.get()
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    text = res.get("text", "").strip()
                    if text:
                        text = text.replace(" ", "")
                        print(f"识别到语音: {text}")
                        # 计算语音向量
                        voice_vec = embedder.encode(text, normalize_embeddings=True)
                        # 发送给主进程
                        msg_queue.put(("voice", (text, voice_vec)))
    except Exception as e:
        msg_queue.put(("error", f"音频循环出错: {e}"))

# ================= 辅助绘图函数 =================

def ensure_cat_image():
    if not os.path.exists("hajimi1.png"):
        img = np.zeros((100, 100, 4), dtype=np.uint8)
        cv2.circle(img, (50, 50), 40, (0, 255, 255, 255), -1)
        cv2.circle(img, (35, 40), 5, (0, 0, 0, 255), -1)
        cv2.circle(img, (65, 40), 5, (0, 0, 0, 255), -1)
        cv2.ellipse(img, (50, 60), (10, 5), 0, 0, 180, (0, 0, 0, 255), 2)
        cv2.imwrite("hajimi1.png", img)

def overlay_img(background, overlay, x, y):
    h, w = overlay.shape[:2]
    if x < 0 or y < 0 or x + w > background.shape[1] or y + h > background.shape[0]:
        return

    # 如果没有 alpha 通道，直接覆盖
    if overlay.shape[2] == 3:
        background[y:y+h, x:x+w] = overlay
        return

    # 有 alpha 通道（BGRA）
    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        background[y:y+h, x:x+w, c] = (
            alpha * overlay[:, :, c] +
            (1 - alpha) * background[y:y+h, x:x+w, c]
        )

def draw_text_chinese(img, text, position, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    draw = ImageDraw.Draw(img)
    # 检查字体文件是否存在，不存在则使用默认
    try:
        font = ImageFont.truetype(FONT_PATH, textSize, encoding="utf-8")
    except:
        font = ImageFont.load_default()
        print(f"警告：无法加载字体 {FONT_PATH}，使用默认字体")
        
    draw.text(position, text, textColor, font=font)
    
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def draw_cat(frame, cat_img, target_box=None):
    """
    绘制哈基米。
    如果 target_box 不为 None，则画箭头指向目标。
    否则仅在画面中间显示哈基米。
    """
    if cat_img is None: return
    
    # 放大哈基米 (3倍大小)
    scale = 1
    new_w = int(cat_img.shape[1] * scale)
    new_h = int(cat_img.shape[0] * scale)
    cat_resized = cv2.resize(cat_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    h, w, _ = frame.shape
    cat_h, cat_w = cat_resized.shape[:2]
    
    # 居中位置
    pos_x = (w - cat_w) // 2
    pos_y = (h - cat_h) // 2
    
    overlay_img(frame, cat_resized, pos_x, pos_y)
    
    if target_box is not None:
        x1, y1, x2, y2 = target_box
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        # 箭头从哈基米中心发出
        start_pt = (pos_x + cat_w // 2, pos_y + cat_h // 2)
        cv2.arrowedLine(frame, start_pt, (cx, cy), (0, 255, 0), 3, tipLength=0.1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)


# ================= Modern GUI App =================

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class HajimiApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # 窗口设置
        self.title("Hajimi AI Assistant")
        self.geometry("1280x800")
        
        # 布局配置
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # --- 左侧边栏 (日志与状态) ---
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(2, weight=1)
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Hajimi AI", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="状态: 初始化中...", text_color="gray")
        self.status_label.grid(row=1, column=0, padx=20, pady=10)
        
        self.log_textbox = ctk.CTkTextbox(self.sidebar_frame, width=200)
        self.log_textbox.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        # --- 设置按钮 ---
        self.settings_btn = ctk.CTkButton(self.sidebar_frame, text="⚙️ 设置模型路径", 
                                          fg_color="transparent", border_width=1,
                                          command=self.open_settings)
        self.settings_btn.grid(row=3, column=0, padx=20, pady=10)
        
        # --- 底部：作者链接 ---
        self.link_label = ctk.CTkLabel(self.sidebar_frame, text="By Fz2hOpensource Team", 
                                       font=ctk.CTkFont(size=12, underline=True),
                                       text_color="lightblue", cursor="hand2")
        self.link_label.grid(row=4, column=0, pady=20)
        self.link_label.bind("<Button-1>", lambda e: webbrowser.open(USER_HOMEPAGE))
        
        # --- 右侧主区域 (视频流) ---
        self.video_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.video_frame.grid(row=0, column=1, sticky="nsew")
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="", corner_radius=10)
        self.video_label.pack(fill="both", expand=True, padx=20, pady=20)
        
        # --- 底部指令显示 ---
        self.command_label = ctk.CTkLabel(self.video_frame, text="等待语音指令...", 
                                          font=ctk.CTkFont(size=20),
                                          fg_color=("white", "gray20"), corner_radius=8)
        self.command_label.place(relx=0.5, rely=0.9, anchor="center")

        # --- 内部状态 ---
        self.cap = None
        self.process = None
        self.msg_queue = None
        self.yolo_model = None
        self.known_vectors = {}
        self.last_voice_vector = None
        self.last_voice_text = ""
        self.cat_img = None
        
        # 延迟初始化，确保 GUI 先显示
        self.after(100, self.start_system)

    def log(self, message):
        self.log_textbox.insert("end", message + "\n")
        self.log_textbox.see("end")

    def open_settings(self):
        """打开设置窗口"""
        settings_window = ctk.CTkToplevel(self)
        settings_window.title("系统设置")
        settings_window.geometry("600x400")
        settings_window.grab_set()  # 模态窗口

        # 标题
        ctk.CTkLabel(settings_window, text="模型路径设置", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=20)

        # 表单容器
        form_frame = ctk.CTkFrame(settings_window)
        form_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # 1. Vosk 路径
        ctk.CTkLabel(form_frame, text="Vosk 语音模型路径 (文件夹):").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        vosk_entry = ctk.CTkEntry(form_frame, width=300)
        vosk_entry.grid(row=1, column=0, padx=10, pady=5)
        vosk_entry.insert(0, VOSK_MODEL_PATH)
        
        def browse_vosk():
            path = filedialog.askdirectory(initialdir=BASE_DIR, title="选择 Vosk 模型文件夹")
            if path:
                vosk_entry.delete(0, "end")
                vosk_entry.insert(0, path)
        
        ctk.CTkButton(form_frame, text="浏览", width=60, command=browse_vosk).grid(row=1, column=1, padx=10)

        # 2. YOLO 路径
        ctk.CTkLabel(form_frame, text="YOLO 模型路径 (.pt 文件):").grid(row=2, column=0, sticky="w", padx=10, pady=(20, 5))
        yolo_entry = ctk.CTkEntry(form_frame, width=300)
        yolo_entry.grid(row=3, column=0, padx=10, pady=5)
        yolo_entry.insert(0, YOLO_MODEL_NAME)
        
        def browse_yolo():
            path = filedialog.askopenfilename(initialdir=BASE_DIR, title="选择 YOLO 模型文件", filetypes=[("YOLO Model", "*.pt")])
            if path:
                yolo_entry.delete(0, "end")
                yolo_entry.insert(0, path)
        
        ctk.CTkButton(form_frame, text="浏览", width=60, command=browse_yolo).grid(row=3, column=1, padx=10)

        # 保存按钮
        def save_and_close():
            new_config = {
                "vosk_path": vosk_entry.get(),
                "yolo_path": yolo_entry.get()
            }
            save_config(new_config)
            tk_msg = "配置已保存！\n请重启程序以生效。"
            # 简单的弹窗提示 (这里用 label 模拟，或者 print)
            print(tk_msg)
            settings_window.destroy()
            self.log("配置已更新，请重启程序。")

        ctk.CTkButton(settings_window, text="保存设置", command=save_and_close, fg_color="green").pack(pady=20)

    def start_system(self):
        self.log("正在启动系统...")
        ensure_cat_image()
        self.cat_img = cv2.imread("hajimi1.png", cv2.IMREAD_UNCHANGED)
        
        # 1. 启动子进程
        self.msg_queue = mp.Queue()
        self.process = mp.Process(target=voice_process_run, 
                                  args=(self.msg_queue, EN_ZH_CACHE, VOSK_MODEL_PATH, EMBEDDING_MODEL_NAME))
        self.process.start()
        self.log("子进程已启动，正在加载语音模型...")
        
        # 2. 加载 YOLO (这可能会卡顿一下 UI，实际生产可以用线程加载)
        self.log("正在加载 YOLO 模型...")
        # 为了不完全卡死 UI，使用 after 稍微分步
        self.after(100, self.load_yolo)

    def load_yolo(self):
        try:
            self.yolo_model = YOLO(YOLO_MODEL_NAME)
            self.log("YOLO 模型加载完成")
        except Exception as e:
            self.log(f"YOLO 加载失败: {e}")
            return
            
        # 3. 打开摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.log("无法打开摄像头")
            return
            
        self.status_label.configure(text="状态: 运行中", text_color="green")
        self.log("系统启动完成！")
        
        # 开始循环
        self.update_loop()

    def update_loop(self):
        if not self.cap or not self.cap.isOpened():
            return

        # 1. 读取摄像头
        ret, frame = self.cap.read()
        if not ret:
            self.log("无法读取视频帧")
            return
            
        frame = cv2.flip(frame, 1)

        # 2. 处理消息队列
        while not self.msg_queue.empty():
            try:
                msg_type, data = self.msg_queue.get_nowait()
                if msg_type == "voice":
                    text, vec = data
                    self.log(f"收到指令: {text}")
                    self.last_voice_text = text
                    self.last_voice_vector = vec
                    self.command_label.configure(text=f"指令: {text}")
                elif msg_type == "init_vectors":
                    self.known_vectors = data
                    self.log(f"已接收 {len(self.known_vectors)} 个类别的向量数据")
                elif msg_type == "error":
                    self.log(f"[Error] {data}")
                elif msg_type == "log":
                    self.log(data)
            except:
                break

        # 3. YOLO 检测
        if self.yolo_model:
            results = self.yolo_model(frame, verbose=False)[0]
            scene_objects = []

            for box in results.boxes:
                cls_id = int(box.cls[0])
                en_name = self.yolo_model.names[cls_id]
                zh_name = EN_ZH_CACHE.get(en_name, en_name)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                scene_objects.append({"zh": zh_name, "box": (x1, y1, x2, y2)})

            # 4. 匹配逻辑
            best_idx = -1
            best_score = 0.3 # 阈值
            
            if self.last_voice_vector is not None and self.known_vectors:
                for i, obj in enumerate(scene_objects):
                    if obj["zh"] in self.known_vectors:
                        obj_vec = self.known_vectors[obj["zh"]]
                        score = np.dot(obj_vec, self.last_voice_vector)
                        if score > best_score:
                            best_score = score
                            best_idx = i

            # 5. 绘制
            for i, obj in enumerate(scene_objects):
                x1, y1, x2, y2 = obj["box"]
                color = (200, 200, 200)
                if i == best_idx: color = (0, 255, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # 使用 PIL 绘制中文
                frame = draw_text_chinese(frame, obj["zh"], (x1, y1 - 30), color, 20)

            if best_idx != -1:
                draw_cat(frame, self.cat_img, scene_objects[best_idx]["box"])
            else:
                draw_cat(frame, self.cat_img)

        # 6. 显示到 Tkinter
        # OpenCV BGR -> RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # 调整大小以适应窗口 (可选，这里简单缩放)
        # 获取 label 的当前大小
        # w = self.video_label.winfo_width()
        # h = self.video_label.winfo_height()
        # if w > 10 and h > 10:
        #    img_pil = img_pil.resize((w, h), Image.Resampling.LANCZOS)
        
        # 转换为 CTkImage
        ctk_img = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=img_pil.size)
        self.video_label.configure(image=ctk_img)
        self.video_label.image = ctk_img # 防止垃圾回收

        # 循环
        self.after(10, self.update_loop)

    def on_closing(self):
        if self.cap:
            self.cap.release()
        if self.process:
            self.process.terminate()
        self.destroy()

if __name__ == "__main__":
    # 必须调用，防止 Windows 下多进程出错
    mp.freeze_support()
    
    app = HajimiApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
    

    def on_closing(self):
        if self.cap:
            self.cap.release()
        if self.process:
            self.process.terminate()
        self.destroy()