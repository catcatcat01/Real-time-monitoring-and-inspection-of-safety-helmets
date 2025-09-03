import threading
import json
import os
import base64
import time
import cv2
import numpy as np
from io import BytesIO
from flask import Blueprint, render_template, request, jsonify, current_app, Response
from PIL import Image
from datetime import timedelta
from app.models.superbrain_client import SuperBrainClient
from app.models.detector import HardHatDetector
from app.utils.alert import send_feishu_alert

superbrain_bp = Blueprint("superbrain", __name__)


# 工具函数：转换为JSON可序列化类型
def convert_to_json_serializable(obj):
    import numpy as np
    from datetime import datetime, timedelta
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (datetime, timedelta)):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(v) for v in obj]
    else:
        try:
            json.dumps(obj)
            return obj
        except:
            return str(obj)


# 检测器单例
detector_instance = None
detector_lock = threading.Lock()


def get_detector():
    global detector_instance
    with detector_lock:
        if detector_instance is None:
            detector_instance = HardHatDetector()
        return detector_instance


@superbrain_bp.route("/superbrain/monitor")
def superbrain_monitor():
    """超脑监控页面入口"""
    return render_template("superbrain_monitor.html")


@superbrain_bp.route("/api/superbrain/stream-detect")
def stream_detect():
    """SSE接口：多设备超脑实时流检测"""
    app = current_app._get_current_object()

    # 获取设备参数（默认设备1）
    try:
        device = int(request.args.get('device', 1))
        if device not in [1, 2]:
            raise ValueError("设备ID必须为1或2")
    except ValueError as e:
        return Response(
            f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n",
            mimetype="text/event-stream"
        )

    def generate_with_context():
        with app.app_context():
            # 1. 初始化客户端和检测器
            try:
                sb_client = SuperBrainClient.get_instance(app=app, device=device)
                detector = get_detector()
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'初始化失败: {str(e)}'})}\n\n"
                return

            # 2. 初始化SDK
            if not sb_client.init_sdk():
                yield f"data: {json.dumps({'type': 'error', 'message': f'设备{device}SDK初始化失败'})}\n\n"
                return

            # 3. 登录设备（增加重试机制）
            login_success = False
            for _ in range(3):  # 最多重试3次
                if sb_client.login():
                    login_success = True
                    break
                time.sleep(1)  # 重试间隔1秒

            if not login_success:
                yield f"data: {json.dumps({'type': 'error', 'message': f'设备{device}登录失败'})}\n\n"
                return

            # 4. 启动预览
            if not sb_client.start_preview():
                sb_client.logout()
                yield f"data: {json.dumps({'type': 'error', 'message': f'设备{device}预览启动失败'})}\n\n"
                return

            # 5. 初始化参数
            fps = 25
            current_frame = 0
            skip_frames = int(round(fps / 5))  # 每5帧处理1次
            monitor_info = {
                "location": sb_client.monitor_location,
                "device_ip": sb_client.ip,
                "device_id": device,
                "channel": sb_client.current_channel
            }

            try:
                while True:
                    # 获取解码帧
                    frame = sb_client.get_latest_frame()
                    if frame is None or frame.size == 0:
                        current_frame += 1
                        time.sleep(0.01)
                        continue

                    # 跳过部分帧
                    if current_frame % skip_frames != 0:
                        current_frame += 1
                        time.sleep(1 / fps)
                        continue

                    # 6. 核心检测逻辑
                    timestamp = current_frame / fps
                    time_str = str(timedelta(seconds=timestamp)).split(".")[0]

                    # 6.1 安全帽检测
                    hat_result = detector.detect_hard_hat(frame)
                    hat_result = convert_to_json_serializable(hat_result)
                    unauthorized_boxes = [d for d in hat_result["detections"] if d["class_id"] == 0]

                    # 6.2 人脸识别
                    faces = []
                    if hat_result["has_unauthorized"]:
                        faces = detector.detect_and_recognize_faces_in_unauthorized(frame, unauthorized_boxes)
                        faces = convert_to_json_serializable(faces)

                    # 7. 绘制检测框
                    frame_with_boxes = frame.copy()
                    # 7.1 绘制安全帽框
                    for det in hat_result["detections"]:
                        x1, y1, x2, y2 = map(int, det["box"])
                        box_color = (0, 0, 255) if det["class_id"] == 0 else (0, 255, 0)
                        text_color = (255, 0, 0) if det["class_id"] == 0 else (0, 255, 0)
                        cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), box_color, 2)
                        text = f"{det['class_name']}: {det['confidence']:.2f}"
                        frame_with_boxes = detector.draw_chinese_text(frame_with_boxes, text, (x1, y1 - 20), text_color,
                                                                      18)

                    # 7.2 绘制人脸框
                    for face in faces:
                        x1, y1, x2, y2 = map(int, face["box"])
                        box_color = (255, 255, 0) if face["name"] != "未知人员" else (255, 0, 0)
                        text_color = (0, 255, 255) if face["name"] != "未知人员" else (0, 0, 255)
                        cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), box_color, 2)
                        text = f"{face['name']}({face['code']})" if face["name"] != "未知人员" else "未知人员"
                        frame_with_boxes = detector.draw_chinese_text(frame_with_boxes, text, (x1, y2 + 20), text_color,
                                                                      18)
                        q_text = f"质量:{face['quality']:.2f} 相似度:{face['similarity']:.2f}"
                        frame_with_boxes = detector.draw_chinese_text(frame_with_boxes, q_text, (x1, y2 + 50),
                                                                      text_color, 16)

                    # 7.3 添加监控信息
                    frame_with_boxes = detector.draw_chinese_text(
                        frame_with_boxes, f"区域: {monitor_info['location']}", (10, 30), (0, 0, 255), 18
                    )
                    frame_with_boxes = detector.draw_chinese_text(
                        frame_with_boxes, f"时间: {time_str}", (10, 70), (0, 0, 255), 18
                    )
                    frame_with_boxes = detector.draw_chinese_text(
                        frame_with_boxes, f"设备{device} 通道: {sb_client.current_channel}", (10, 110), (0, 0, 255), 18
                    )
                    frame_with_boxes = detector.draw_chinese_text(
                        frame_with_boxes, f"未佩戴人数: {hat_result['unauthorized_count']}", (10, 150), (0, 0, 255), 18
                    )

                    # 8. 违规告警
                    if hat_result["has_unauthorized"] and hat_result["unauthorized_count"] > 0:
                        alert_data = {
                            **monitor_info,
                            "unauthorized_count": hat_result["unauthorized_count"],
                            "total_detections": hat_result["total_detections"],
                            "time_str": time_str,
                            "alert_type": f"超脑设备{device}监控违规"
                        }
                        send_feishu_alert(alert_data, frame_with_boxes)

                    # 9. 帧转Base64
                    encode_start = time.time()
                    ret, buffer = cv2.imencode('.jpg', frame_with_boxes, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    if not ret:
                        app.logger.error(f"设备{device}编码失败")
                        continue
                    frame_base64 = base64.b64encode(buffer).decode("utf-8")
                    encode_cost = time.time() - encode_start
                    frame_data = f"data:image/jpeg;base64,{frame_base64}"
                    app.logger.debug(f"设备{device}编码耗时: {encode_cost:.3f}秒")

                    # 10. 推送数据 - 确保每次都返回totalChannels
                    send_data = {
                        "type": "frame",
                        "frameData": frame_data,
                        "timeStr": time_str,
                        "currentChannel": sb_client.current_channel,
                        "totalChannels": sb_client.total_channels,  # 确保每次都返回总通道数
                        "unauthorizedCount": hat_result["unauthorized_count"],
                        "faces": faces
                    }
                    try:
                        yield f"data: {json.dumps(send_data)}\n\n"
                    except Exception as e:
                        app.logger.error(f"设备{device}数据序列化失败: {str(e)}")
                        send_data = {
                            "type": "frame",
                            "frameData": frame_data,
                            "timeStr": time_str,
                            "currentChannel": sb_client.current_channel,
                            "totalChannels": sb_client.total_channels,
                            "unauthorizedCount": 0,
                            "faces": [],
                            "error": "数据序列化失败"
                        }
                        yield f"data: {json.dumps(send_data)}\n\n"

                    current_frame += 1
                    time.sleep(1 / fps)

            except Exception as e:
                err_msg = f"设备{device}检测错误: {str(e)}"
                current_app.logger.error(err_msg, exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'message': err_msg})}\n\n"

            finally:
                sb_client.logout()
                yield f"data: {json.dumps({'type': 'end', 'message': f'设备{device}监控已停止'})}\n\n"

    # SSE响应配置
    return Response(
        generate_with_context(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@superbrain_bp.route("/api/superbrain/switch-channel", methods=["POST"])
def switch_channel():
    """切换指定设备的通道"""
    data = request.json
    try:
        device = int(data.get("device", 1))
        target_channel = int(data.get("channel", 1))
        if device not in [1, 2]:
            return jsonify({"status": "error", "message": "设备ID必须为1或2"}), 400
        if not isinstance(target_channel, int) or target_channel < 1:
            return jsonify({"status": "error", "message": "通道号必须为正整数"}), 400
    except ValueError:
        return jsonify({"status": "error", "message": "参数格式错误"}), 400

    # 切换通道
    app = current_app._get_current_object()
    try:
        # 确保获取正确的设备实例
        sb_client = SuperBrainClient.get_instance(app=app, device=device)

        # 检查登录状态，如未登录则尝试重新登录
        if sb_client.user_id < 0:
            app.logger.warning(f"设备{device}未登录，尝试重新登录...")
            if not sb_client.login():
                return jsonify({"status": "error", "message": f"设备{device}未登录，重新登录失败"}), 400

        # 执行通道切换
        if sb_client.switch_channel(target_channel):
            return jsonify({
                "status": "success",
                "currentChannel": sb_client.current_channel,
                "totalChannels": sb_client.total_channels,  # 返回总通道数
                "message": f"设备{device}已切换到通道 {sb_client.current_channel}"
            })
        else:
            return jsonify({
                "status": "error",
                "message": f"设备{device}切换通道 {target_channel} 失败，通道号可能超出范围",
                "totalChannels": sb_client.total_channels  # 返回总通道数供前端参考
            }), 500
    except Exception as e:
        app.logger.error(f"设备{device}切换通道异常: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": f"操作失败: {str(e)}"}), 500


@superbrain_bp.route("/api/superbrain/release-device", methods=["POST"])
def release_device():
    """主动释放指定设备的所有资源"""
    app = current_app._get_current_object()
    data = request.json
    try:
        device = int(data.get("device", 1))
        if device not in [1, 2]:
            return jsonify({"status": "error", "message": "设备ID必须为1或2"}), 400
    except ValueError:
        return jsonify({"status": "error", "message": "参数格式错误"}), 400

    try:
        # 获取设备实例并释放资源
        key = str(device)
        if key in SuperBrainClient._clients:
            sb_client = SuperBrainClient._clients[key]
            sb_client.logout()
            del SuperBrainClient._clients[key]
            app.logger.info(f"主动释放设备{device}资源完成")
            return jsonify({"status": "success", "message": f"设备{device}资源已释放"})
        else:
            app.logger.warning(f"设备{device}实例不存在，无需释放")
            return jsonify({"status": "success", "message": f"设备{device}实例不存在"})
    except Exception as e:
        app.logger.error(f"释放设备{device}资源异常: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": f"释放失败: {str(e)}"}), 500


@superbrain_bp.route("/api/superbrain/get-channels", methods=["POST"])
def get_channels():
    """新增接口：主动获取设备通道信息"""
    app = current_app._get_current_object()
    data = request.json
    try:
        device = int(data.get("device", 1))
        if device not in [1, 2]:
            return jsonify({"status": "error", "message": "设备ID必须为1或2"}), 400
    except ValueError:
        return jsonify({"status": "error", "message": "参数格式错误"}), 400

    try:
        # 获取设备实例
        sb_client = SuperBrainClient.get_instance(app=app, device=device)

        # 确保设备已登录
        if sb_client.user_id < 0:
            app.logger.info(f"获取通道信息前，设备{device}未登录，尝试登录...")
            if not sb_client.login():
                return jsonify({"status": "error", "message": f"设备{device}登录失败，无法获取通道信息"}), 400

        # 返回通道信息
        return jsonify({
            "status": "success",
            "totalChannels": sb_client.total_channels,
            "currentChannel": sb_client.current_channel,
            "message": f"成功获取设备{device}通道信息"
        })
    except Exception as e:
        app.logger.error(f"获取设备{device}通道信息异常: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": f"获取失败: {str(e)}"}), 500
