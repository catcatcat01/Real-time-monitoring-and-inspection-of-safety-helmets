from flask import Flask
from config import Config
from app.models.db import init_db
from app.routes import main_bp, detection_bp


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # 初始化数据库
    init_db(app)

    # 注册蓝图
    app.register_blueprint(main_bp)
    app.register_blueprint(detection_bp, url_prefix='/detection')

    # 延迟导入并注册 superbrain_bp
    from app.routes.superbrain import superbrain_bp
    app.register_blueprint(superbrain_bp)  # 新增

    return app