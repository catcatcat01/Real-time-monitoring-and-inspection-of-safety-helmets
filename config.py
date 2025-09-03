import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # 数据库配置
    DB_HOST = os.getenv('DB_HOST', '10.66.102.120')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_USER = os.getenv('DB_USER', 'postgres')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '123456')
    DB_NAME = os.getenv('DB_NAME', 'fastgate')

    # 应用配置
    UPLOAD_FOLDER = os.path.join('app', 'static', 'uploads')
    RESULTS_FOLDER = os.path.join('app', 'static', 'results')
    FACE_CACHE_DIR = os.path.join('FaceImages', 'cache')
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

    # 模型路径
    HAT_MODEL_PATH = os.path.join('models', 'yolo11n_safehat.pt')
    FACE_MODEL_PATH = os.path.join('models', 'yolov8n-face.pt')
    SHAPE_PREDICTOR_PATH = os.path.join('models', 'shape_predictor_68_face_landmarks.dat')
    FACE_REC_MODEL_PATH = os.path.join('models', 'dlib_face_recognition_resnet_model_v1.dat')

    # 实时监控m3u8地址（用户提供的链接）
    MONITOR_M3U8_URL = "https://open.ys7.com/v3/openlive/FK9284610_1_1.m3u8?expire=1786764266&id=878238042844680192&t=14b64c503c51412bf12bb0ac93bbe423ff302e6f196a81587d25836c7eb689ab&ev=100"
    # 监控区域名称（用于告警标识）
    MONITOR_LOCATION = "M4-1F-化成"

    # 确保目录存在
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    # -------------------------- 超脑设备1配置 --------------------------
    SUPERBRAIN1_IP = "10.60.201.249"  # 设备1IP
    SUPERBRAIN1_PORT = 8000  # 设备1端口（默认8000）
    SUPERBRAIN1_USER = "admin"  # 设备1用户名
    SUPERBRAIN1_PWD = "waffer248"  # 设备1密码
    SUPERBRAIN1_SDK_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'lib')  # 设备1SDK路径
    SUPERBRAIN1_LOG_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'logs', 'superbrain1')  # 设备1日志路径
    SUPERBRAIN1_MONITOR_LOCATION = "WIM超脑监控"  # 设备1监控区域名

    # -------------------------- 超脑设备2配置 --------------------------
    SUPERBRAIN2_IP = "10.62.3.222"  # 设备2IP（替换为实际IP）
    SUPERBRAIN2_PORT = 8000  # 设备2端口
    SUPERBRAIN2_USER = "admin"  # 设备2用户名
    SUPERBRAIN2_PWD = "waffer12"  # 设备2密码（替换为实际密码）
    SUPERBRAIN2_SDK_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'lib')  # 设备2SDK路径（可与设备1共用）
    SUPERBRAIN2_LOG_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'logs', 'superbrain2')  # 设备2日志路径
    SUPERBRAIN2_MONITOR_LOCATION = "WIH超脑监控"  # 设备2监控区域名

    os.makedirs(SUPERBRAIN1_LOG_PATH, exist_ok=True)
    os.makedirs(SUPERBRAIN2_LOG_PATH, exist_ok=True)