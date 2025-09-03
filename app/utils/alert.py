import time
import json
import requests
import os
from datetime import datetime
from flask import current_app
import cv2
import numpy as np

# 飞书应用配置
FEISHU_APP_ID = "cli_a8e897b12f27500e"  # 替换为你的飞书应用ID
FEISHU_APP_SECRET = "bQtAu5D4DuECIa4t5zK0je3QNqK4e5m8"  # 替换为你的飞书应用密钥
FEISHU_RECEIVE_USER_IDS = ["10407843"]  # 接收告警的用户ID列表
ALERT_COOLDOWN = 60000  # 告警冷却时间(秒)

# 缓存访问令牌和过期时间
feishu_token_cache = {
    "token": None,
    "expire_time": 0
}

# 最近告警时间记录
recent_alert_times = {}


def get_feishu_access_token():
    """获取并缓存飞书访问令牌"""
    # 检查缓存中是否有未过期的令牌
    if feishu_token_cache["token"] and time.time() < feishu_token_cache["expire_time"]:
        return feishu_token_cache["token"]

    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    headers = {"Content-Type": "application/json"}
    data = {
        "app_id": FEISHU_APP_ID,
        "app_secret": FEISHU_APP_SECRET
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        token_data = response.json()
        # 缓存令牌，提前5分钟过期
        feishu_token_cache["token"] = token_data.get("tenant_access_token")
        feishu_token_cache["expire_time"] = time.time() + token_data.get("expire", 7200) - 300
        return feishu_token_cache["token"]
    except Exception as e:
        current_app.logger.error(f"获取飞书访问令牌失败: {e}")
        return None


def send_feishu_alert(detection_result, frame_image=None):
    """发送安全帽检测告警到飞书

    Args:
        detection_result: 检测结果字典
        frame_image: 当前帧的图像数据（numpy数组）
    """
    # 检查冷却时间
    video_id = detection_result.get('video_id', 'unknown')
    current_time = time.time()
    last_alert_time = recent_alert_times.get(video_id, 0)

    if current_time - last_alert_time < ALERT_COOLDOWN:
        current_app.logger.info(f"告警冷却中，跳过发送: {video_id}")
        return False

    access_token = get_feishu_access_token()
    if not access_token:
        current_app.logger.error("无法获取飞书访问令牌，无法发送告警")
        return False

    # 如果提供了帧图像，则保存到本地以便通过URL访问
    image_url = None
    if frame_image is not None and isinstance(frame_image, np.ndarray):
        try:
            # 创建文件夹保存告警图片
            alert_images_dir = os.path.join(current_app.config.get('RESULTS_FOLDER', 'results'), 'alert_frames')
            os.makedirs(alert_images_dir, exist_ok=True)

            # 生成唯一的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            image_filename = f"alert_frame_{timestamp}.jpg"
            image_path = os.path.join(alert_images_dir, image_filename)

            # 保存图像到本地
            success = cv2.imwrite(image_path, frame_image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if success:
                # 生成可以通过Web访问的URL
                base_url = current_app.config.get('BASE_URL', 'http://10.60.208.45:5000/')
                # 构造相对于RESULTS_FOLDER的路径
                relative_path = os.path.relpath(image_path, current_app.config.get('RESULTS_FOLDER', 'results'))
                image_url = f"{base_url}detection/results/{relative_path.replace(os.sep, '/')}"
                current_app.logger.info(f"告警图像已保存: {image_path}")
            else:
                current_app.logger.error(f"保存图像到本地失败: {image_path}")
        except Exception as e:
            current_app.logger.error(f"处理告警图像时出错: {e}")

    # 构建详情链接
    base_url = current_app.config.get('BASE_URL', 'http://10.60.208.45:5000/')
    detail_url = ""
    if 'filename' in detection_result and 'output_filename' in detection_result:
        detail_url = f"{base_url}/detection/process/{detection_result.get('filename')}?output={detection_result.get('output_filename')}"

    # 格式化告警时间
    alert_time = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")

    # 构建告警内容
    elements = [
        {
            "tag": "div",
            "text": {
                "content": f"**视频名称**: {detection_result.get('filename', '未知')}\n"
                           f"**检测时间**: {alert_time}\n"
                           f"**未佩戴人数**: {detection_result.get('unauthorized_count', 0)}\n"
                           f"**总检测人数**: {detection_result.get('total_detections', 0)}",
                "tag": "lark_md"
            }
        },
        {
            "tag": "div",
            "text": {
                "content": f"**告警位置**: {detection_result.get('location', '未知区域')}",
                "tag": "lark_md"
            }
        }
    ]

    # 如果有图像URL，添加查看图像的按钮
    actions = []
    if image_url:
        actions.append({
            "tag": "button",
            "text": {
                "content": "查看违规截图",
                "tag": "plain_text"
            },
            "type": "primary",
            "multi_url": {
                "url": image_url,
                "pc_url": image_url,
                "android_url": image_url,
                "ios_url": image_url
            }
        })

    # 添加详情链接按钮（如果可用）
    if detail_url:
        actions.append({
            "tag": "button",
            "text": {
                "content": "查看详细记录",
                "tag": "plain_text"
            },
            "type": "default",
            "multi_url": {
                "url": detail_url,
                "pc_url": detail_url,
                "android_url": detail_url,
                "ios_url": detail_url
            }
        })

    if actions:
        elements.append({
            "tag": "action",
            "actions": actions
        })

    alert_content = {
        "msg_type": "interactive",
        "card": {
            "config": {
                "wide_screen_mode": True
            },
            "header": {
                "title": {
                    "content": "⚠️ 未佩戴安全帽检测告警",
                    "tag": "plain_text"
                },
                "template": "red"
            },
            "elements": elements
        }
    }

    # 发送给所有指定用户
    success_count = 0
    for user_id in FEISHU_RECEIVE_USER_IDS:
        if _send_to_single_user(access_token, user_id, alert_content):
            success_count += 1

    # 更新最近告警时间
    if success_count > 0:
        recent_alert_times[video_id] = current_time
        return True
    return False


def _send_to_single_user(access_token, user_id, content):
    """向单个用户发送飞书消息"""
    url = "https://open.feishu.cn/open-apis/im/v1/messages"
    params = {"receive_id_type": "user_id"}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    data = {
        "receive_id": user_id,
        "content": json.dumps(content["card"]),
        "msg_type": "interactive"
    }

    try:
        response = requests.post(url, params=params, headers=headers, json=data)
        if response.status_code == 200:
            current_app.logger.info(f"飞书告警发送成功，用户: {user_id}")
            return True
        current_app.logger.error(
            f"飞书告警发送失败，用户: {user_id}, 状态码: {response.status_code}, 响应: {response.text}")
        return False
    except Exception as e:
        current_app.logger.error(f"发送飞书告警时出错，用户: {user_id}, 错误: {e}")
        return False
