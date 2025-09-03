# 8.14 ： 识别能力优化版（带本地缓存和并行处理）

import cv2
import time
import os
import torch
import numpy as np
from datetime import timedelta
from ultralytics import YOLO
import requests
from io import BytesIO
from flask import current_app
from app.models.db import get_known_faces
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw, ImageFont
import warnings
import threading
import hashlib
import concurrent.futures
import pickle
from collections import defaultdict
import traceback

warnings.filterwarnings("ignore", category=FutureWarning)


class FaceCache:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.cache_dir = current_app.config.get('FACE_CACHE_DIR', 'face_cache')
            os.makedirs(self.cache_dir, exist_ok=True)
            self.cache_file = os.path.join(self.cache_dir, 'face_encodings_cache.pkl')
            self.known_face_encodings = []
            self.known_face_names = []
            self.known_face_codes = []
            self.load_cache()

    def get_image_cache_path(self, name, code, img_url, index=0):
        """获取图片缓存路径，格式为name_code_index.jpg"""
        # 清理name和code中的特殊字符
        clean_name = "".join(c for c in name if c.isalnum() or c in (' ', '_')).strip().replace(' ', '_')
        clean_code = "".join(c for c in code if c.isalnum() or c in (' ', '_')).strip().replace(' ', '_')

        # 使用URL哈希确保唯一性
        url_hash = hashlib.md5(img_url.encode()).hexdigest()[:8]
        return os.path.join(self.cache_dir, f"{clean_name}_{clean_code}_{index}_{url_hash}.jpg")

    def load_cache(self):
        """从缓存文件加载已知人脸数据"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data.get('encodings', [])
                    self.known_face_names = data.get('names', [])
                    self.known_face_codes = data.get('codes', [])
                current_app.logger.info(f"从缓存加载 {len(self.known_face_encodings)} 个人脸特征")
            except Exception as e:
                current_app.logger.error(f"加载缓存失败: {str(e)}\n{traceback.format_exc()}")
                # 加载失败时清除无效缓存
                try:
                    os.remove(self.cache_file)
                except:
                    pass
                self.known_face_encodings = []
                self.known_face_names = []
                self.known_face_codes = []

    def save_cache(self):
        """保存已知人脸数据到缓存文件"""
        data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names,
            'codes': self.known_face_codes
        }
        try:
            # 先写入临时文件，再重命名，避免写入过程中出错
            temp_file = self.cache_file + '.tmp'
            with open(temp_file, 'wb') as f:
                pickle.dump(data, f)

            # 确保目录存在
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

            # 替换原文件
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
            os.rename(temp_file, self.cache_file)

            current_app.logger.info(f"成功保存 {len(self.known_face_encodings)} 个人脸特征到缓存")
            return True
        except Exception as e:
            current_app.logger.error(f"保存缓存失败: {str(e)}\n{traceback.format_exc()}")
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
            return False

    def clear_cache(self):
        """清除缓存"""
        try:
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
            self.known_face_encodings = []
            self.known_face_names = []
            self.known_face_codes = []
            current_app.logger.info("人脸缓存已清除")
        except Exception as e:
            current_app.logger.error(f"清除缓存失败: {str(e)}")


class HardHatDetector:
    def __init__(self):
        """初始化带人脸识别的安全帽检测器"""
        # 设备选择
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载YOLO模型
        self.hat_model = YOLO(current_app.config['HAT_MODEL_PATH'])
        self.hat_model.to(self.device)

        # 初始化InsightFace分析器
        self.face_analyzer = FaceAnalysis(
            name='buffalo_l',
            root='models/',
            providers=['CUDAExecutionProvider' if self.device == 'cuda' else 'CPUExecutionProvider'],
            allowed_modules=['detection', 'recognition'],
            download=False
        )
        self.face_analyzer.prepare(ctx_id=0 if self.device == 'cuda' else -1)

        # 模型参数
        self.input_size = (640, 640)
        self.face_match_threshold = 0.35
        self.min_face_confidence = 0.5

        # 获取单例缓存
        self.face_cache = FaceCache.get_instance()

        # 如果缓存为空，从数据库加载
        if not self.face_cache.known_face_encodings:
            self.load_known_faces_from_db()

        # 加载字体
        self.font_path = "./fonts/SimHei.ttf"
        if not os.path.exists(self.font_path):
            raise FileNotFoundError(f"字体文件不存在: {self.font_path}")



    def normalize_embedding(self, embedding):
        """归一化特征向量"""
        if embedding is None:
            return None
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    def calculate_face_quality(self, face):
        """计算人脸质量分数"""
        if not hasattr(face, 'kps') or face.kps is None:
            return face.det_score if hasattr(face, 'det_score') else 0.0

        try:
            kps = face.kps
            if len(kps) < 5:
                return face.det_score if hasattr(face, 'det_score') else 0.0

            # 5点模型：左眼、右眼、鼻子、左嘴角、右嘴角
            left_eye = kps[0]
            right_eye = kps[1]
            nose = kps[2]
            mouth_left = kps[3]
            mouth_right = kps[4]

            eye_distance = np.linalg.norm(left_eye - right_eye)
            eyes_center = (left_eye + right_eye) / 2
            nose_to_eyes = np.linalg.norm(nose - eyes_center)
            mouth_width = np.linalg.norm(mouth_left - mouth_right)

            if eye_distance > 0 and nose_to_eyes > 0:
                quality_score = min(1.0, eye_distance / nose_to_eyes * 0.5 + mouth_width / eye_distance * 0.5)
            else:
                quality_score = 0.5

            return quality_score * (face.det_score if hasattr(face, 'det_score') else 0.5)
        except Exception as e:
            current_app.logger.error(f"计算人脸质量失败: {str(e)}")
            return face.det_score if hasattr(face, 'det_score') else 0.5

    def download_and_cache_image(self, img_url, name, code, index):
        """下载并缓存图片"""
        cache_path = self.face_cache.get_image_cache_path(name, code, img_url, index)

        # 如果图片已缓存，直接返回路径
        if os.path.exists(cache_path):
            return cache_path

        try:
            response = requests.get(img_url, timeout=10)
            response.raise_for_status()

            # 保存图片到缓存目录
            with open(cache_path, 'wb') as f:
                f.write(response.content)

            return cache_path
        except Exception as e:
            current_app.logger.error(f"下载图片失败: {img_url}, 错误: {str(e)}")
            return None

    def process_single_face(self, face_data, index):
        """处理单个人脸数据"""
        try:
            # 下载或获取缓存图片
            cache_path = self.download_and_cache_image(
                face_data['img_path'],
                face_data['name'],
                face_data['code'],
                index
            )

            if not cache_path or not os.path.exists(cache_path):
                current_app.logger.warning(f"图片缓存失败: {face_data['name']}({face_data['code']})")
                return None

            # 读取图片
            img = Image.open(cache_path)
            img = img.convert('RGB')
            rgb_image = np.array(img)

            # 检测人脸
            faces = self.face_analyzer.get(rgb_image)
            if not faces:
                current_app.logger.warning(f"未检测到人脸: {face_data['name']}({face_data['code']})")
                return None

            # 获取最佳人脸
            best_face = max(faces, key=lambda x: x.det_score)
            if best_face.det_score < self.min_face_confidence:
                current_app.logger.warning(
                    f"跳过低质量人脸({best_face.det_score:.2f}): {face_data['name']}({face_data['code']})")
                return None

            # 计算质量分数
            quality_score = self.calculate_face_quality(best_face)

            # 归一化特征向量
            normalized_embedding = self.normalize_embedding(best_face.embedding)
            if normalized_embedding is None:
                current_app.logger.warning(f"无法获取特征向量: {face_data['name']}({face_data['code']})")
                return None

            return {
                'embedding': normalized_embedding,
                'name': face_data['name'],
                'code': face_data['code'],
                'det_score': best_face.det_score,
                'quality_score': quality_score,
                'keypoints': best_face.kps.tolist() if hasattr(best_face, 'kps') and best_face.kps is not None else []
            }
        except Exception as e:
            current_app.logger.error(f"处理人脸 {face_data['name']} 失败: {str(e)}\n{traceback.format_exc()}")
            return None

    def load_known_faces_from_db(self):
        """从数据库加载已知人脸（使用并行处理）"""
        current_app.logger.info("开始从数据库加载已知人脸")
        print("开始从数据库加载已知人脸")

        # 获取人脸数据并按人员分组
        faces_data = get_known_faces()
        if not faces_data:
            current_app.logger.warning("未从数据库获取到人脸数据")
            return

        # 按人员分组，每人最多两张照片
        grouped_faces = defaultdict(list)
        for face in faces_data:
            grouped_faces[(face['name'], face['code'])].append(face)

        # 准备处理任务（每人最多两张照片）
        tasks = []
        for (name, code), faces in grouped_faces.items():
            for i, face in enumerate(faces[:3]):  # 每人最多处理两张照片
                tasks.append((face, i))  # i作为照片索引

        # 使用线程池并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(tasks))) as executor:
            futures = [executor.submit(self.process_single_face, task[0], task[1]) for task in tasks]

            loaded_count = 0
            # 添加同步等待所有任务完成
            for future in concurrent.futures.as_completed(futures):
                try:
                    face_info = future.result()
                    if face_info:
                        self.face_cache.known_face_encodings.append(face_info)
                        loaded_count += 1

                        # 输出日志
                        print(f"加载人脸: {face_info['name']}({face_info['code']}), "
                              f"质量: {face_info['quality_score']:.2f}, "
                              f"特征(前3): {face_info['embedding'][:3] if face_info['embedding'] is not None else 'N/A'}")

                        current_app.logger.info(
                            f"加载人脸: {face_info['name']}({face_info['code']}), "
                            f"质量: {face_info['quality_score']:.2f}, "
                            f"特征(前3): {face_info['embedding'][:3] if face_info['embedding'] is not None else 'N/A'}"
                        )
                except Exception as e:
                    current_app.logger.error(f"处理人脸失败: {str(e)}")

        # 确保所有任务完成后才保存缓存
        self.face_cache.save_cache()

        # 打印加载完成提示
        current_app.logger.info(f"成功加载 {loaded_count} 个高质量人脸数据")
        print(f"成功加载 {loaded_count} 个高质量人脸数据")

        # 确保所有特征加载完成后再继续
        return loaded_count > 0  # 返回是否成功加载了人脸数据


    def detect_hard_hat(self, frame):
        """检测单帧中的安全帽佩戴情况"""
        results = self.hat_model(frame, imgsz=self.input_size, verbose=False)

        hat_detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # 假设0: 未佩戴安全帽, 1: 佩戴安全帽（请根据实际模型类别调整）
                hat_detections.append({
                    'class_id': cls_id,
                    'class_name': "未佩戴安全帽" if cls_id == 0 else "佩戴安全帽",
                    'confidence': confidence,
                    'box': box.xyxy[0].tolist()  # 边界框坐标
                })

        # 判断是否有未佩戴安全帽的情况
        has_unauthorized = any(d['class_id'] == 0 for d in hat_detections)

        return {
            'has_unauthorized': has_unauthorized,
            'total_detections': len(hat_detections),
            'unauthorized_count': sum(1 for d in hat_detections if d['class_id'] == 0),
            'authorized_count': sum(1 for d in hat_detections if d['class_id'] == 1),
            'detections': hat_detections
        }

    def detect_and_recognize_faces_in_unauthorized(self, frame, unauthorized_boxes):
        """仅在未佩戴安全帽区域内检测并识别人脸"""
        faces = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 使用 InsightFace 检测所有人脸
        results = self.face_analyzer.get(rgb_frame)
        if results is None:
            return faces

        for face in results:
            # 确保bbox有效
            if not hasattr(face, 'bbox') or face.bbox is None:
                continue

            # 跳过低质量人脸
            if face.det_score < self.min_face_confidence:
                continue

            x1, y1, x2, y2 = face.bbox.astype(int)

            # 检查是否在未佩戴安全帽区域内
            in_unauthorized_area = False
            for hat_box in unauthorized_boxes:
                fx1, fy1, fx2, fy2 = map(int, hat_box['box'])
                if (x1 < fx2 and x2 > fx1 and y1 < fy2 and y2 > fy1):
                    in_unauthorized_area = True
                    break

            if not in_unauthorized_area:
                continue

            # 确保embedding有效
            if not hasattr(face, 'embedding') or face.embedding is None:
                continue

            # 计算人脸质量分数
            quality_score = self.calculate_face_quality(face)

            # 归一化当前人脸特征
            current_embedding = self.normalize_embedding(face.embedding)

            # 特征比对（使用余弦相似度）
            name = "未知人员"
            code = ""
            best_match = None
            min_distance = float('inf')
            similarity = 0.0

            if self.face_cache.known_face_encodings and current_embedding is not None:
                # 提取已知特征向量
                known_embeddings = [f['embedding'] for f in self.face_cache.known_face_encodings if f['embedding'] is not None]
                known_faces = self.face_cache.known_face_encodings

                # 计算余弦相似度
                if known_embeddings and current_embedding is not None:
                    similarities = np.dot(known_embeddings, current_embedding)
                    best_match_idx = np.argmax(similarities)
                    max_similarity = similarities[best_match_idx]
                    min_distance = 1 - max_similarity  # 转换为距离

                    # 获取最佳匹配
                    best_match = known_faces[best_match_idx]

                    # 仅使用相似度作为主要匹配标准
                    if max_similarity > self.face_match_threshold:
                        name = best_match['name']
                        code = best_match['code']
                        similarity = max_similarity

            # 提取关键点（安全处理）
            keypoints = []
            if hasattr(face, 'kps') and face.kps is not None:
                # 限制关键点数量以避免索引错误
                keypoints = face.kps.tolist()[:min(len(face.kps), 68)]

            faces.append({
                'box': [x1, y1, x2, y2],
                'confidence': face.det_score if hasattr(face, 'det_score') else 0.0,
                'quality': quality_score,
                'name': name,
                'code': code,
                'distance': min_distance,
                'similarity': similarity,
                'matched_info': best_match,
                'keypoints': keypoints
            })

        return faces

    def draw_chinese_text(self, image, text, position, color=(0, 255, 0), font_size=20):
        """
        在OpenCV图像上绘制中文文本
        :param image: OpenCV图像 (BGR格式)
        :param text: 要绘制的中文文本
        :param position: (x, y) 文本左下角坐标
        :param color: (B, G, R) 颜色值
        :param font_size: 字体大小
        :return: 绘制后的图像
        """

        # 转换颜色通道 (BGR -> RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)

        # 加载字体
        try:
            font = ImageFont.truetype(self.font_path, font_size)
        except:
            font = ImageFont.load_default()
            print("警告：使用默认字体，可能不支持中文")

        # 绘制文本
        draw.text(position, text, font=font, fill=color)

        # 转换回OpenCV格式
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def process_video(self, video_path, output_filename):
        """处理视频文件，检测安全帽佩戴情况并识别相关人员"""
        # 创建保存结果的目录
        output_dir = os.path.join(current_app.config['RESULTS_FOLDER'], output_filename)
        os.makedirs(output_dir, exist_ok=True)

        frames_dir = os.path.join(output_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)

        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / fps  # 视频总时长(秒)

        # 计算每隔多少帧取一帧（每秒一帧）
        frame_interval = 1
        # 计算每隔多少帧抽取一次（根据视频帧率）
        skip_frames = int(round(fps / frame_interval))
        current_app.logger.info(f"视频帧率: {fps:.2f}, 总帧数: {total_frames}, 总时长: {duration_seconds:.2f}秒")
        current_app.logger.info(f"帧抽取间隔: 每{skip_frames}帧抽取1帧（确保1秒1帧）")

        # 初始化视频写入器
        output_video_path = os.path.join(output_dir, 'result.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, frame_interval, (width, height))

        results = []
        frame_count = 0
        saved_frame_count = 0
        start_time = time.time()

        current_app.logger.info(f"开始处理视频: {video_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # 视频结束

            # 关键修改：只处理间隔帧（1秒1帧）
            if frame_count % skip_frames != 0:
                frame_count += 1
                continue  # 跳过不需要处理的帧

            # 处理帧
            timestamp = frame_count / fps
            time_str = str(timedelta(seconds=timestamp)).replace(':', '-')
            hat_result = self.detect_hard_hat(frame)

            # 提取未佩戴安全帽的区域
            unauthorized_boxes = [d for d in hat_result['detections'] if d['class_id'] == 0]

            # 只在检测到未佩戴安全帽时才进行人脸检测
            faces = []
            if hat_result['has_unauthorized'] and unauthorized_boxes:
                # 只检测未佩戴安全帽区域内的人脸
                faces = self.detect_and_recognize_faces_in_unauthorized(frame, unauthorized_boxes)

            hat_result['faces'] = faces
            hat_result['timestamp'] = timestamp
            hat_result['time_str'] = str(timedelta(seconds=timestamp))
            hat_result['frame_count'] = frame_count  # 记录当前处理的帧序号
            results.append(hat_result)

            # 绘制边界框
            frame_with_boxes = frame.copy()

            # 绘制安全帽检测框
            for detection in hat_result['detections']:
                box = detection['box']
                x1, y1, x2, y2 = map(int, box)
                confidence = detection['confidence']
                class_name = detection['class_name']

                # 未佩戴用红色，佩戴用绿色
                if detection['class_id'] == 0:  # 未佩戴安全帽
                    box_color = (0, 0, 255)  # 红色框
                    text_color = (255, 0, 0)  # 红色文字
                else:  # 佩戴安全帽
                    box_color = (0, 255, 0)  # 绿色框
                    text_color = (0, 255, 0)  # 绿色文字

                cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), box_color, 2)

                text = f"{class_name}: {confidence:.2f}"
                frame_with_boxes = self.draw_chinese_text(
                    frame_with_boxes,
                    text,
                    position=(x1, y1 - 20),
                    color=text_color,
                    font_size=18
                )

            # 绘制未佩戴安全帽人员的人脸框
            if hat_result['has_unauthorized']:
                for face in faces:
                    x1, y1, x2, y2 = map(int, face['box'])
                    name = face['name']
                    code = face['code']

                    # 已知人脸用青色框，未知用蓝色框
                    if face['name'] != "未知人员":  # 已知人员
                        box_color = (255, 255, 0)  # 青色框
                        text_color = (0, 255, 255)  # 青色文字
                    else:  # 未知人员
                        box_color = (255, 0, 0)  # 蓝色框
                        text_color = (0, 0, 255)  # 蓝色文字

                    cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), box_color, 2)

                    # 安全绘制关键点（仅当存在时）
                    if face['keypoints']:
                        for kp in face['keypoints']:
                            if len(kp) >= 2:
                                x, y = int(kp[0]), int(kp[1])
                                cv2.circle(frame_with_boxes, (x, y), 2, (0, 255, 255), -1)

                    # 替换原来的cv2.putText为中文绘制
                    display_text = f"{name}({code})" if name != "未知人员" else "未知人员"
                    frame_with_boxes = self.draw_chinese_text(
                        frame_with_boxes,
                        display_text,
                        position=(x1, y2 + 20),
                        color=text_color,
                        font_size=18
                    )

                    # 显示质量分数和相似度
                    quality_text = f"质量:{face['quality']:.2f} 相似度:{face['similarity']:.2f}"
                    frame_with_boxes = self.draw_chinese_text(
                        frame_with_boxes,
                        quality_text,
                        position=(x1, y2 + 50),
                        color=text_color,
                        font_size=16
                    )

            # 添加时间戳
            cv2.putText(frame_with_boxes, f"Time: {hat_result['time_str']}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2)

            # 添加未佩戴安全帽计数
            cv2.putText(frame_with_boxes, f"Unauthorized: {hat_result['unauthorized_count']}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 2)

            # 保存检测到未佩戴安全帽的帧
            if hat_result['has_unauthorized']:
                saved_frame_count += 1
                frame_filename = f"{frames_dir}/unauthorized_{time_str}_frame_{frame_count}.jpg"
                cv2.imwrite(frame_filename, frame_with_boxes)

            # 写入视频
            out.write(frame_with_boxes)
            frame_count += 1

        # 释放资源
        cap.release()
        out.release()

        # 计算处理时间
        processing_time = time.time() - start_time
        current_app.logger.info(f"视频处理完成，总时长: {str(timedelta(seconds=total_frames / fps))}")
        current_app.logger.info(f"处理时间: {processing_time:.2f}秒")

        return {
            'output_path': output_video_path,
            'frames_dir': frames_dir,
            'results': results,
            'fps': fps,
            'processing_time': processing_time,
            'total_frames': total_frames,
            'unauthorized_frames': sum(1 for r in results if r['has_unauthorized'])
        }
