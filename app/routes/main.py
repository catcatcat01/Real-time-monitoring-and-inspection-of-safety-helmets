from flask import Blueprint, render_template, redirect, url_for, request, flash
import os
import uuid
from app.models.detector import HardHatDetector

main_bp = Blueprint('main', __name__)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@main_bp.route('/')
def index():
    return render_template('index.html')


@main_bp.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # 检查是否有文件上传
        if 'video' not in request.files:
            flash('未选择视频文件')
            return redirect(request.url)

        file = request.files['video']

        # 检查文件名是否为空
        if file.filename == '':
            flash('未选择视频文件')
            return redirect(request.url)

        # 检查文件类型
        if file and allowed_file(file.filename):
            # 生成唯一文件名
            filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
            upload_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                       '..', 'static', 'uploads', filename)

            # 保存文件
            file.save(upload_path)

            # 重定向到处理页面
            return redirect(url_for('detection.process', filename=filename))

    return render_template('upload.html')

@main_bp.route('/monitor')
def monitor():
    # 重定向到detection蓝图中的监控页面（复用已实现的监控功能）
    return redirect(url_for('detection.monitor_page'))