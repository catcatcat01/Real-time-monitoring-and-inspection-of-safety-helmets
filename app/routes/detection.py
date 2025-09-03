import os
import json
import base64
import time
import pickle
import threading
from io import BytesIO
from flask import Blueprint, render_template, request, jsonify, current_app, send_from_directory, url_for, Response
from app.models.detector import HardHatDetector
import uuid
from urllib.parse import quote
from PIL import Image
import cv2
from datetime import timedelta
from app.utils.alert import send_feishu_alert

detection_bp = Blueprint('detection', __name__)

# -------------------------- 新增1：Detector单例（避免重复初始化模型） --------------------------
detector_instance = None
detector_lock = threading.Lock()


def get_detector():
    """获取HardHatDetector单例（线程安全）"""
    global detector_instance
    with detector_lock:
        if detector_instance is None:
            detector_instance = HardHatDetector()
        return detector_instance


# -------------------------- 新增2：异步上传接口（供上传页AJAX调用） --------------------------
@detection_bp.route('/api/upload', methods=['POST'])
def api_upload():
    """上传视频并返回唯一文件名"""
    if 'video' not in request.files:
        return jsonify({'status': 'error', 'error': '未选择视频文件'}), 400

    file = request.files['video']
    if not file.filename:
        return jsonify({'status': 'error', 'error': '未选择视频文件'}), 400

    # 生成唯一文件名并保存
    ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'mp4'
    filename = str(uuid.uuid4()) + '.' + ext
    upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(upload_path)
        current_app.logger.info(f"视频上传成功: {filename}")
        return jsonify({'status': 'success', 'filename': filename})
    except Exception as e:
        current_app.logger.error(f"视频保存失败: {str(e)}")
        return jsonify({'status': 'error', 'error': '视频保存失败'}), 500


# -------------------------- 新增3：实时检测页路由（渲染video_detect.html） --------------------------
@detection_bp.route('/video-detect/<filename>')
def video_detect(filename):
    """实时检测页面入口"""
    video_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(video_path):
        return "视频文件不存在", 404
    return render_template('video_detect.html', filename=filename)


# -------------------------- 新增4：SSE推流接口（核心：实时处理帧并推送） --------------------------
@detection_bp.route('/api/stream-detect/<filename>')
def stream_detect(filename):
    """SSE接口：逐帧处理视频，推送检测后的帧数据"""
    app = current_app._get_current_object()

    # 转换函数保持不变
    def convert_to_json_serializable(obj):
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                              np.int16, np.int32, np.int64, np.uint8,
                              np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        else:
            return obj

    def generate_with_context():
        with app.app_context():
            try:
                video_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                if not os.path.exists(video_path):
                    yield f"data: {json.dumps({'type': 'error', 'message': '视频文件不存在'})}\n\n"
                    return

                detector = get_detector()
                output_filename = str(uuid.uuid4())
                output_dir = os.path.join(current_app.config['RESULTS_FOLDER'], output_filename)
                os.makedirs(output_dir, exist_ok=True)
                frames_dir = os.path.join(output_dir, 'frames')
                os.makedirs(frames_dir, exist_ok=True)

                # 打开视频
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    yield f"data: {json.dumps({'type': 'error', 'message': '无法打开视频'})}\n\n"
                    return
                fps = cap.get(cv2.CAP_PROP_FPS)
                skip_frames = int(round(fps / 5))
                current_frame = 0
                results = []
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                processing_start_time = time.time()

                # 初始化视频写入器（实时生成结果视频，避免二次处理）
                output_video_path = os.path.join(output_dir, 'result.mp4')
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_video_path, fourcc, 8, (width, height))  # 8fps与检测速度匹配

                # 新增：记录视频基本信息
                video_info = {
                    'video_id': output_filename,
                    'filename': filename,
                    'output_filename': output_filename,
                    'location': current_app.config.get('DETECTION_LOCATION', '默认区域')  # 可在配置中设置监控区域
                }

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if current_frame % skip_frames != 0:
                        current_frame += 1
                        continue

                    # 帧处理逻辑
                    timestamp = current_frame / fps
                    time_str = str(timedelta(seconds=timestamp)).split('.')[0]

                    # 检测安全帽
                    hat_result = detector.detect_hard_hat(frame)
                    hat_result = convert_to_json_serializable(hat_result)
                    unauthorized_boxes = [d for d in hat_result['detections'] if d['class_id'] == 0]

                    # 检测人脸
                    faces = []
                    if hat_result['has_unauthorized']:
                        faces = detector.detect_and_recognize_faces_in_unauthorized(frame, unauthorized_boxes)
                        faces = convert_to_json_serializable(faces)

                    # 绘制检测框
                    frame_with_boxes = frame.copy()
                    # 绘制安全帽框
                    for det in hat_result['detections']:
                        x1, y1, x2, y2 = map(int, det['box'])
                        if det['class_id'] == 0:  # 未佩戴安全帽
                            box_color = (0, 0, 255)  # 红色框
                            text_color = (255, 0, 0)  # 红色文字
                        else:  # 佩戴安全帽
                            box_color = (0, 255, 0)  # 绿色框
                            text_color = (0, 255, 0)  # 绿色文字
                        #color = (0, 0, 255) if det['class_id'] == 0 else (0, 255, 0)
                        cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), box_color, 2)
                        text = f"{det['class_name']}: {det['confidence']:.2f}"
                        frame_with_boxes = detector.draw_chinese_text(frame_with_boxes, text, (x1, y1 - 20), text_color, 18)

                    # 绘制人脸框
                    for face in faces:
                        x1, y1, x2, y2 = map(int, face['box'])
                        # 已知人脸用青色框，未知用蓝色框
                        if face['name'] != "未知人员":  # 已知人员
                            box_color = (255, 255, 0)  # 青色框
                            text_color = (0, 255, 255)  # 青色文字
                        else:  # 未知人员
                            box_color = (255, 0, 0)  # 蓝色框
                            text_color = (0, 0, 255)  # 蓝色文字
                        #color = (255, 255, 0) if face['name'] != '未知人员' else (255, 0, 0)
                        cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), box_color, 2)
                        text = f"{face['name']}({face['code']})" if face['name'] != '未知人员' else '未知人员'
                        frame_with_boxes = detector.draw_chinese_text(frame_with_boxes, text, (x1, y2 + 20), text_color, 18)
                        q_text = f"质量:{face['quality']:.2f} 相似度:{face['similarity']:.2f}"
                        frame_with_boxes = detector.draw_chinese_text(frame_with_boxes, q_text, (x1, y2 + 50), text_color,
                                                                      16)

                    # 添加时间戳和统计信息
                    cv2.putText(frame_with_boxes, f"Time: {time_str}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (0, 255, 0), 2)
                    cv2.putText(frame_with_boxes, f"Unauthorized: {hat_result['unauthorized_count']}",
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (0, 0, 255), 2)

                    # 写入结果视频（实时生成，避免二次处理）
                    out.write(frame_with_boxes)

                    # 保存违规帧（实时检测时就保存，结果页直接读取）
                    if hat_result['has_unauthorized']:
                        frame_filename = f"unauthorized_{time_str.replace(':', '-')}_frame_{current_frame}.jpg"
                        frame_path = os.path.join(frames_dir, frame_filename)
                        cv2.imwrite(frame_path, frame_with_boxes)

                    # 检测到未佩戴安全帽时发送告警
                    if hat_result['has_unauthorized'] and hat_result['unauthorized_count'] > 0:
                        # 构建告警数据
                        alert_data = {
                            **video_info,
                            'unauthorized_count': hat_result['unauthorized_count'],
                            'total_detections': hat_result['total_detections'],
                            'frame_count': current_frame,
                            'time_str': time_str
                        }
                        # 发送告警（冷却机制在工具函数内部处理）
                        send_feishu_alert(alert_data)

                    # 帧转Base64推送到前端
                    frame_rgb = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    buf = BytesIO()
                    pil_img.save(buf, format='JPEG', quality=85)
                    frame_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                    frame_data = f"data:image/jpeg;base64,{frame_base64}"

                    # 收集结果
                    hat_result.update({
                        'faces': faces,
                        'timestamp': timestamp,
                        'time_str': time_str,
                        'frame_count': current_frame
                    })
                    safe_result = convert_to_json_serializable(hat_result)
                    results.append(safe_result)

                    # 推送数据
                    send_data = {
                        'type': 'frame',
                        'frameData': frame_data,
                        'timeStr': time_str,
                        'totalDetections': safe_result['total_detections'],
                        'unauthorizedCount': safe_result['unauthorized_count'],
                        'authorizedCount': safe_result['authorized_count'],
                        'faces': safe_result['faces']
                    }
                    yield f"data: {json.dumps(send_data)}\n\n"

                    time.sleep(1 / fps)
                    current_frame += 1

                # 关键修复1：关闭视频写入器，不调用process_video二次检测
                out.release()
                processing_time = time.time() - processing_start_time

                # 保存结果数据 - 确保结构正确
                final_result = {
                    'results': results,
                    'fps': fps,
                    'total_frames': total_frames,
                    'processing_time': processing_time,
                    'output_path': output_video_path,
                    'frames_dir': frames_dir,
                    'unauthorized_frames': sum(1 for r in results if r['has_unauthorized'])
                }

                # 确保所有数据都是JSON可序列化的
                safe_final_result = convert_to_json_serializable(final_result)

                with open(os.path.join(output_dir, 'results.pkl'), 'wb') as f:
                    pickle.dump(safe_final_result, f)

                # 推送结束信号
                yield f"data: {json.dumps({'type': 'end','outputFilename': output_filename,'message': '检测完成'})}\n\n"

            except Exception as e:
                err_msg = str(e)
                current_app.logger.error(f"SSE检测错误: {err_msg}")
                yield f"data: {json.dumps({'type': 'error', 'message': err_msg})}\n\n"

            finally:
                if 'cap' in locals() and cap.isOpened():
                    cap.release()
                if 'out' in locals():
                    out.release()


    return Response(
        generate_with_context(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@detection_bp.route('/process/<filename>')
def process(filename):
    """结果页：读取实时检测生成的结果，不重新处理"""
    output_filename = request.args.get('output')
    if not output_filename:
        return "缺少检测结果标识", 400

    video_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    output_dir = os.path.join(current_app.config['RESULTS_FOLDER'], output_filename)
    results_path = os.path.join(output_dir, 'results.pkl')

    try:
        if not os.path.exists(results_path):
            return "检测结果不存在", 404

        # 加载结果并验证结构
        with open(results_path, 'rb') as f:
            result = pickle.load(f)

        # 验证结果结构是否正确
        required_keys = ['results', 'fps', 'output_path', 'frames_dir', 'unauthorized_frames']
        if not all(key in result for key in required_keys):
            return "检测结果格式错误", 500

        # 确保results是列表
        if not isinstance(result['results'], list):
            result['results'] = []

        # 构建URL
        video_rel_path = os.path.relpath(result['output_path'], current_app.config['RESULTS_FOLDER']).replace(os.sep,
                                                                                                              '/')
        frames_rel_dir = os.path.relpath(result['frames_dir'], current_app.config['RESULTS_FOLDER']).replace(os.sep,
                                                                                                             '/')

        video_url = url_for('detection.get_result_file', filename=quote(video_rel_path))
        frames_dir_url = url_for('detection.get_result_file', filename=quote(frames_rel_dir))

        return render_template('results.html',
                               result=result,
                               video_path=video_url,
                               frames_dir=frames_dir_url)

    except Exception as e:
        current_app.logger.error(f"结果页错误: {str(e)}")
        # 打印详细的错误信息和数据结构，帮助调试
        current_app.logger.error(f"结果数据类型: {type(result)}")
        if isinstance(result, dict):
            current_app.logger.error(f"结果数据键: {result.keys()}")
        return "加载结果出错，请查看日志", 500


@detection_bp.route('/results/<path:filename>')
def get_result_file(filename):
    """提供结果文件（视频/图片）的访问"""
    # 结果文件的根目录（即RESULTS_FOLDER）
    results_root = current_app.config['RESULTS_FOLDER']
    # 安全检查：确保请求的文件在结果目录内
    safe_path = os.path.abspath(os.path.join(results_root, filename))
    if not safe_path.startswith(os.path.abspath(results_root)):
        return "访问被拒绝", 403

    # 获取目录和文件名
    directory, file_name = os.path.split(safe_path)
    return send_from_directory(directory, file_name)

@detection_bp.route('/api/process', methods=['POST'])
def api_process():
    """API接口处理视频"""
    if 'video' not in request.files:
        return jsonify({'error': '未选择视频文件'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': '未选择视频文件'}), 400

    # 保存文件
    filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
    upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)

    try:
        # 处理视频
        detector = HardHatDetector()
        output_filename = str(uuid.uuid4())
        result = detector.process_video(upload_path, output_filename)

        # 准备返回数据
        return jsonify({
            'status': 'success',
            'processing_time': result['processing_time'],
            'total_frames': result['total_frames'],
            'unauthorized_frames': result['unauthorized_frames'],
            'results': result['results']
        })

    except Exception as e:
        current_app.logger.error(f"API视频处理错误: {str(e)}")
        return jsonify({'error': str(e)}), 500




# -------------------------- 新增5：实时监控页面路由 --------------------------
@detection_bp.route('/monitor')
def monitor_page():
    """实时监控页面入口（渲染监控画面）"""
    # 从配置获取监控地址，若未配置则使用默认值
    monitor_url = current_app.config.get('MONITOR_M3U8_URL')
    if not monitor_url:
        return "未配置监控流地址", 400
    return render_template('monitor.html', monitor_location=current_app.config['MONITOR_LOCATION'])


# -------------------------- 新增6：实时监控SSE推流接口 --------------------------
@detection_bp.route('/api/stream-monitor')
def stream_monitor():
    """SSE接口：拉取m3u8监控流，实时处理并推送检测结果"""
    app = current_app._get_current_object()

    # 复用原有JSON序列化函数（避免numpy类型报错）
    def convert_to_json_serializable(obj):
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                              np.int16, np.int32, np.int64, np.uint8,
                              np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        else:
            return obj

    def generate_with_context():
        with app.app_context():
            cap = None
            try:
                # 1. 获取监控流地址并初始化流读取
                monitor_url = current_app.config.get('MONITOR_M3U8_URL')
                cap = cv2.VideoCapture(monitor_url)
                if not cap.isOpened():
                    err_msg = "无法连接到监控流，请检查地址或网络"
                    current_app.logger.error(err_msg)
                    yield f"data: {json.dumps({'type': 'error', 'message': err_msg})}\n\n"
                    return

                # 2. 初始化检测器（复用单例）
                detector = get_detector()
                fps = cap.get(cv2.CAP_PROP_FPS) or 25  # 若流无FPS信息，默认25帧
                skip_frames = int(round(fps / 5))  # 每5帧处理1次（平衡性能与实时性）
                current_frame = 0
                monitor_info = {
                    'monitor_location': current_app.config['MONITOR_LOCATION'],
                    'stream_url': monitor_url
                }

                current_app.logger.info(f"开始监控：{monitor_info['monitor_location']}，流地址：{monitor_url}")

                # 3. 循环拉取并处理帧
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        err_msg = "监控流中断，正在重试..."
                        current_app.logger.warning(err_msg)
                        yield f"data: {json.dumps({'type': 'warning', 'message': err_msg})}\n\n"
                        # 重试连接（等待3秒后重新初始化）
                        time.sleep(3)
                        cap = cv2.VideoCapture(monitor_url)
                        if not cap.isOpened():
                            continue
                        continue

                    # 跳过部分帧，降低处理压力
                    if current_frame % skip_frames != 0:
                        current_frame += 1
                        continue

                    # 4. 核心检测逻辑（复用现有函数）
                    timestamp = current_frame / fps
                    time_str = str(timedelta(seconds=timestamp)).split('.')[0]

                    # 4.1 检测安全帽
                    hat_result = detector.detect_hard_hat(frame)
                    hat_result = convert_to_json_serializable(hat_result)
                    unauthorized_boxes = [d for d in hat_result['detections'] if d['class_id'] == 0]

                    # 4.2 仅在未佩戴区域识别人脸
                    faces = []
                    if hat_result['has_unauthorized']:
                        faces = detector.detect_and_recognize_faces_in_unauthorized(frame, unauthorized_boxes)
                        faces = convert_to_json_serializable(faces)

                    # 5. 绘制检测框（复用现有绘图逻辑）
                    frame_with_boxes = frame.copy()
                    # 绘制安全帽框（红=未佩戴，绿=佩戴）
                    for det in hat_result['detections']:
                        x1, y1, x2, y2 = map(int, det['box'])
                        box_color = (0, 0, 255) if det['class_id'] == 0 else (0, 255, 0)
                        text_color = (255, 0, 0) if det['class_id'] == 0 else (0, 255, 0)
                        cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), box_color, 2)
                        text = f"{det['class_name']}: {det['confidence']:.2f}"
                        frame_with_boxes = detector.draw_chinese_text(frame_with_boxes, text, (x1, y1 - 20), text_color, 18)

                    # 绘制人脸框（青=已知，蓝=未知）
                    for face in faces:
                        x1, y1, x2, y2 = map(int, face['box'])
                        box_color = (255, 255, 0) if face['name'] != "未知人员" else (255, 0, 0)
                        text_color = (0, 255, 255) if face['name'] != "未知人员" else (0, 0, 255)
                        cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), box_color, 2)
                        text = f"{face['name']}({face['code']})" if face['name'] != "未知人员" else "未知人员"
                        frame_with_boxes = detector.draw_chinese_text(frame_with_boxes, text, (x1, y2 + 20), text_color, 18)
                        q_text = f"质量:{face['quality']:.2f} 相似度:{face['similarity']:.2f}"
                        frame_with_boxes = detector.draw_chinese_text(frame_with_boxes, q_text, (x1, y2 + 50), text_color, 16)

                    # -------------------------- 修正：用draw_chinese_text绘制中文（替换原3行cv2.putText） --------------------------
                    # 1. 绘制监控区域
                    frame_with_boxes = detector.draw_chinese_text(
                        image=frame_with_boxes,
                        text=f"监控区域: {monitor_info['monitor_location']}",
                        position=(10, 30),
                        color=(0, 0, 255),
                        font_size=18
                    )

                    # 2. 绘制时间
                    frame_with_boxes = detector.draw_chinese_text(
                        image=frame_with_boxes,
                        text=f"时间: {time_str}",
                        position=(10, 70),
                        color=(0, 0, 255),
                        font_size=18
                    )

                    # 3. 绘制未佩戴人数（红色文字，位置(10,110)，字号18）
                    frame_with_boxes = detector.draw_chinese_text(
                        image=frame_with_boxes,
                        text=f"未佩戴人数: {hat_result['unauthorized_count']}",
                        position=(10, 110),
                        color=(0, 0, 255),  # 原红色（BGR格式）
                        font_size=18
                    )

                    # 6. 检测到违规时发送飞书告警（复用现有告警逻辑）
                    if hat_result['has_unauthorized'] and hat_result['unauthorized_count'] > 0:
                        alert_data = {
                            **monitor_info,
                            'unauthorized_count': hat_result['unauthorized_count'],
                            'total_detections': hat_result['total_detections'],
                            'time_str': time_str,
                            'alert_type': "实时监控违规"
                        }
                        send_feishu_alert(alert_data, frame_with_boxes)  # 冷却机制已在send_feishu_alert内实现

                    # 7. 帧转Base64推送到前端
                    frame_rgb = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    buf = BytesIO()
                    pil_img.save(buf, format='JPEG', quality=80)  # 降低质量减少传输延迟
                    frame_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                    frame_data = f"data:image/jpeg;base64,{frame_base64}"

                    # 8. 推送数据到前端
                    send_data = {
                        'type': 'frame',
                        'frameData': frame_data,
                        'timeStr': time_str,
                        'unauthorizedCount': hat_result['unauthorized_count'],
                        'authorizedCount': hat_result['authorized_count'],
                        'faces': faces,
                        'location': monitor_info['monitor_location']
                    }
                    yield f"data: {json.dumps(send_data)}\n\n"

                    current_frame += 1
                    time.sleep(1 / fps)  # 控制推送频率，避免前端卡顿

            except Exception as e:
                err_msg = f"监控处理错误: {str(e)}"
                current_app.logger.error(err_msg, exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'message': err_msg})}\n\n"

            finally:
                # 释放资源
                if cap and cap.isOpened():
                    cap.release()
                    current_app.logger.info("监控流已关闭")
                yield f"data: {json.dumps({'type': 'end', 'message': '监控已停止'})}\n\n"

    # SSE响应配置（禁用缓存，确保实时性）
    return Response(
        generate_with_context(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',

        }
    )