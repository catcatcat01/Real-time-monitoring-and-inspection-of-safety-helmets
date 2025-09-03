import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # 数据库配置
    DB_HOST = os.getenv('DB_HOST', '')
    DB_PORT = os.getenv('DB_PORT', '')
    DB_USER = os.getenv('DB_USER', '')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    DB_NAME = os.getenv('DB_NAME', '')

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
    MONITOR_M3U8_URL = ""
    # 监控区域名称（用于告警标识）
    MONITOR_LOCATION = ""

    # 确保目录存在
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    # -------------------------- 超脑设备1配置 --------------------------
    SUPERBRAIN1_IP = ""  # 设备1IP
    SUPERBRAIN1_PORT =   # 设备1端口（默认8000）
    SUPERBRAIN1_USER = ""  # 设备1用户名
    SUPERBRAIN1_PWD = ""  # 设备1密码
    SUPERBRAIN1_SDK_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'lib')  # 设备1SDK路径
    SUPERBRAIN1_LOG_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'logs', 'superbrain1')  # 设备1日志路径
    SUPERBRAIN1_MONITOR_LOCATION = ""  # 设备1监控区域名

    # -------------------------- 超脑设备2配置 --------------------------
    SUPERBRAIN2_IP = ""  # 设备2IP（替换为实际IP）
    SUPERBRAIN2_PORT =   # 设备2端口
    SUPERBRAIN2_USER = ""  # 设备2用户名
    SUPERBRAIN2_PWD = ""  # 设备2密码（替换为实际密码）
    SUPERBRAIN2_SDK_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'lib')  # 设备2SDK路径（可与设备1共用）
    SUPERBRAIN2_LOG_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'logs', 'superbrain2')  # 设备2日志路径
    SUPERBRAIN2_MONITOR_LOCATION = ""  # 设备2监控区域名

    os.makedirs(SUPERBRAIN1_LOG_PATH, exist_ok=True)

    os.makedirs(SUPERBRAIN2_LOG_PATH, exist_ok=True)
