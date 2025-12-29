import cv2
import numpy as np
import multiprocessing as mp
import sys
import os
import time
import json
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# ================= 配置区域 =================
VOSK_MODEL_PATH = "vosk-model-small-cn-0.22"
YOLO_MODEL_NAME = "D:\Fun\hajimi\yolo11s.pt"
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
# 中文字体路径 (尝试系统自带的微软雅黑)
FONT_PATH = "C:/Windows/Fonts/msyh.ttc"

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
        
        # 使用 sounddevice 开启流
        with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=None,
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

# ================= 主进程：UI 与 视觉 =================

def ensure_cat_image():
    if not os.path.exists("hajimi.png"):
        img = np.zeros((100, 100, 4), dtype=np.uint8)
        cv2.circle(img, (50, 50), 40, (0, 255, 255, 255), -1)
        cv2.circle(img, (35, 40), 5, (0, 0, 0, 255), -1)
        cv2.circle(img, (65, 40), 5, (0, 0, 0, 255), -1)
        cv2.ellipse(img, (50, 60), (10, 5), 0, 0, 180, (0, 0, 0, 255), 2)
        cv2.imwrite("hajimi.png", img)

def overlay_img(background, overlay, x, y):
    h, w = overlay.shape[:2]
    if x < 0 or y < 0 or x + w > background.shape[1] or y + h > background.shape[0]:
        return
    alpha = overlay[:, :, 3] / 255.0
    for c in range(0, 3):
        background[y:y+h, x:x+w, c] = (alpha * overlay[:, :, c] + 
                                      (1 - alpha) * background[y:y+h, x:x+w, c])

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

def draw_cat_pointing(frame, target_box, cat_img):
    if cat_img is None: return
    h, w, _ = frame.shape
    cat_h, cat_w = cat_img.shape[:2]
    pos_x, pos_y = 20, h - cat_h - 20
    overlay_img(frame, cat_img, pos_x, pos_y)
    
    x1, y1, x2, y2 = target_box
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    start_pt = (pos_x + cat_w // 2, pos_y + cat_h // 2)
    cv2.arrowedLine(frame, start_pt, (cx, cy), (0, 255, 0), 3, tipLength=0.1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

def main():
    # 必须调用，防止 Windows 下多进程出错
    mp.freeze_support()

    ensure_cat_image()
    cat_img = cv2.imread("hajimi.png", cv2.IMREAD_UNCHANGED)

    # 1. 启动子进程
    msg_queue = mp.Queue()
    process = mp.Process(target=voice_process_run, 
                         args=(msg_queue, EN_ZH_CACHE, VOSK_MODEL_PATH, EMBEDDING_MODEL_NAME))
    process.start()

    print("[Main] 等待子进程初始化...")
    
    # 2. 等待初始化向量
    known_vectors = {}
    
    # 简单的非阻塞等待加载 UI 之前
    while True:
        try:
            msg = msg_queue.get(timeout=0.1)
            if msg[0] == "init_vectors":
                known_vectors = msg[1]
                print(f"[Main] 已接收 {len(known_vectors)} 个类别的向量数据")
                break
            elif msg[0] == "error":
                print(f"[Error from Child] {msg[1]}")
                process.terminate()
                return
        except:
            pass
        # 可以在这里打印加载动画，暂时略过

    # 3. 加载 YOLO (主进程)
    print("[Main] 正在加载 YOLO...")
    try:
        yolo_model = YOLO(YOLO_MODEL_NAME)
    except Exception as e:
        print(f"YOLO 加载失败: {e}")
        process.terminate()
        return

    # 4. 主循环
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        process.terminate()
        return

    print("[Main] 系统启动完成！")
    
    last_voice_vector = None
    last_voice_text = ""

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        # 检查消息队列 (非阻塞)
        while not msg_queue.empty():
            try:
                msg_type, data = msg_queue.get_nowait()
                if msg_type == "voice":
                    text, vec = data
                    print(f"[Main] 更新指令: {text}")
                    last_voice_text = text
                    last_voice_vector = vec
                elif msg_type == "error":
                    print(f"[Error] {data}")
                elif msg_type == "log":
                    print(data)
            except:
                break

        # YOLO 检测
        results = yolo_model(frame, verbose=False)[0]
        scene_objects = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            en_name = yolo_model.names[cls_id]
            zh_name = EN_ZH_CACHE.get(en_name, en_name)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            scene_objects.append({"zh": zh_name, "box": (x1, y1, x2, y2)})

        # 匹配逻辑
        best_idx = -1
        best_score = 0.3 # 阈值
        
        if last_voice_vector is not None:
            for i, obj in enumerate(scene_objects):
                # 从预计算的字典里取向量
                if obj["zh"] in known_vectors:
                    obj_vec = known_vectors[obj["zh"]]
                    score = np.dot(obj_vec, last_voice_vector)
                    if score > best_score:
                        best_score = score
                        best_idx = i

        # 绘制
        for i, obj in enumerate(scene_objects):
            x1, y1, x2, y2 = obj["box"]
            color = (200, 200, 200)
            if i == best_idx: color = (0, 255, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # 使用 PIL 绘制中文
            frame = draw_text_chinese(frame, obj["zh"], (x1, y1 - 30), color, 20)

        if best_idx != -1:
            draw_cat_pointing(frame, scene_objects[best_idx]["box"], cat_img)

        # 显示当前指令
        if last_voice_text:
             frame = draw_text_chinese(frame, f"指令: {last_voice_text}", (10, 30), (255, 255, 0), 25)

        cv2.imshow("Hajimi AI (Multi-Process)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    process.terminate()

if __name__ == "__main__":
    main()
