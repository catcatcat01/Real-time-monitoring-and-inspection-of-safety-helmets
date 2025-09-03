from psycopg2.pool import SimpleConnectionPool
from flask import current_app, g

def init_db(app):
    # 初始化数据库连接池
    app.config['DB_POOL'] = SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        host=app.config['DB_HOST'],
        port=app.config['DB_PORT'],
        user=app.config['DB_USER'],
        password=app.config['DB_PASSWORD'],
        database=app.config['DB_NAME']
    )

def get_db():
    if 'db' not in g:
        g.db = current_app.config['DB_POOL'].getconn()
    return g.db

def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        current_app.config['DB_POOL'].putconn(db)

def get_known_faces():
    """从数据库获取已知人脸信息"""
    db = get_db()
    cursor = db.cursor()
    try:
        # 执行用户提供的SQL查询
        cursor.execute("""
            select code, name, 'http://10.66.102.120/pic'||b.person_picture_path imgPath 
            from tbl_person a 
            join tbl_person_picture b on a.code = b.person_code 
            and b.person_picture_path is not null 
            where end_time >= NOW() 
            
        """)
        faces = cursor.fetchall()
        return [
            {
                'code': face[0],
                'name': face[1],
                'img_path': face[2]
            } for face in faces
        ]
    finally:
        cursor.close()