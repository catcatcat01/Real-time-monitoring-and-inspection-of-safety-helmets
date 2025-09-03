import threading
import cv2
import time
import numpy as np
from HCNetSDK import *
from PlayCtrl import *
from threading import Condition
from collections import deque
import os
import sys


class SuperBrainClient:
    """超脑设备客户端（多设备支持版 - 增强资源释放）"""
    _clients = {}  # 多设备实例字典：key=设备ID，value=实例
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, app=None, device=1):
        """按设备ID获取实例（1/2）- 新增前设备资源释放逻辑"""
        with cls._lock:
            key = str(device)
            # 1. 如果当前有其他设备实例，先释放其资源
            for existing_key in list(cls._clients.keys()):
                if existing_key != key:
                    existing_client = cls._clients[existing_key]
                    existing_client.logout()  # 彻底释放前设备资源
                    del cls._clients[existing_key]  # 从缓存中移除
                    app.logger.info(f"已释放前设备{existing_key}的所有资源")

            # 2. 创建/获取当前设备实例
            if key not in cls._clients:
                if app is None:
                    raise ValueError("创建SuperBrainClient必须传入Flask应用实例")
                cls._clients[key] = cls(app, device)
            return cls._clients[key]

    def __init__(self, app, device=1):
        if hasattr(self, "initialized"):
            return
        self.initialized = True
        self.app = app
        self.device = device  # 保存当前设备ID
        self.config_prefix = f"SUPERBRAIN{device}"  # 配置前缀（SUPERBRAIN1/SUPERBRAIN2）

        # -------------------------- 读取当前设备的配置 --------------------------
        self.ip = app.config.get(f"{self.config_prefix}_IP")
        self.port = app.config.get(f"{self.config_prefix}_PORT", 8000)
        self.username = app.config.get(f"{self.config_prefix}_USER")
        self.password = app.config.get(f"{self.config_prefix}_PWD")
        self.sdk_path = app.config.get(f"{self.config_prefix}_SDK_PATH")
        self.log_path = app.config.get(f"{self.config_prefix}_LOG_PATH")
        self.monitor_location = app.config.get(
            f"{self.config_prefix}_MONITOR_LOCATION",
            f"超脑设备{device}"
        )

        # -------------------------- 基础属性初始化 --------------------------
        self.hikSDK = None
        self.playM4SDK = None
        self.user_id = -1  # 登录句柄
        self.real_play_handle = -1  # 预览句柄
        self.play_ctrl_port = C_LONG(-1)  # 播放库端口
        self.current_channel = 1  # 默认通道
        self.total_channels = 0  # 总通道数
        self.device_info = None  # 设备详情

        # 预览回调（裸流回调）
        self.real_data_callback = REALDATACALLBACK(self._real_data_callback)

        # -------------------------- 解码相关属性 --------------------------
        self.decode_lock = Condition()
        self.decode_buf = None
        self.last_frame_time = 0
        self._decode_callback = DECCBFUNWIN(self._decode_callback_impl)
        self.frame_buffer = deque(maxlen=5)

    # -------------------------- 解码回调实现 --------------------------
    def _decode_callback_impl(self, n_port, p_buf, n_size, p_frame_info, n_user, n_reserved2):
        with self.app.app_context():
            try:
                if not p_frame_info or p_frame_info.contents.nType != 3:
                    return

                frame_width = p_frame_info.contents.nWidth
                frame_height = p_frame_info.contents.nHeight
                if frame_width <= 0 or frame_height <= 0:
                    self.app.logger.warning(f"设备{self.device}无效帧尺寸：{frame_width}x{frame_height}")
                    return

                # 获取YUV数据指针并转换
                y_size = frame_width * frame_height
                uv_size = y_size // 4
                yuv_ptr = cast(p_buf, POINTER(c_ubyte))
                base_addr = addressof(yuv_ptr.contents)

                # 读取Y、U、V分量
                y_data = (c_ubyte * y_size).from_address(base_addr)
                u_data = (c_ubyte * uv_size).from_address(base_addr + y_size)
                v_data = (c_ubyte * uv_size).from_address(base_addr + y_size + uv_size)

                # YUV转RGB
                y = np.frombuffer(y_data, dtype=np.uint8).reshape((frame_height, frame_width))
                u = np.frombuffer(u_data, dtype=np.uint8).reshape((frame_height // 2, frame_width // 2))
                v = np.frombuffer(v_data, dtype=np.uint8).reshape((frame_height // 2, frame_width // 2))

                u = cv2.resize(u, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
                v = cv2.resize(v, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)

                y = y.astype(np.float32) - 16
                u = u.astype(np.float32) - 128
                v = v.astype(np.float32) - 128

                r = 1.164 * y + 1.596 * v
                g = 1.164 * y - 0.392 * u - 0.813 * v
                b = 1.164 * y + 2.017 * u

                r = np.clip(r, 0, 255).astype(np.uint8)
                g = np.clip(g, 0, 255).astype(np.uint8)
                b = np.clip(b, 0, 255).astype(np.uint8)

                rgb_frame = np.stack((r, g, b), axis=2)
                bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

                with self.decode_lock:
                    if bgr_frame.size > 0:
                        self.decode_buf = bgr_frame.copy()
                        self.last_frame_time = time.time()
                        self.decode_lock.notify_all()
                        self.app.logger.debug(f"设备{self.device}【解码写帧】通道{self.current_channel}，已通知检测线程")

                self.app.logger.debug(
                    f"设备{self.device}SDK解码成功：通道{self.current_channel}，帧尺寸{frame_width}x{frame_height}")

            except Exception as e:
                self.app.logger.error(f"设备{self.device}SDK解码回调失败: {str(e)}", exc_info=True)

    # -------------------------- SDK加载与初始化 --------------------------
    def _load_sdk(self):
        """加载当前设备的SDK"""
        try:
            sdk_dll_path = os.path.join(self.sdk_path, "HCNetSDK.dll")
            play_dll_path = os.path.join(self.sdk_path, "PlayCtrl.dll")

            if not os.path.exists(sdk_dll_path) or not os.path.exists(play_dll_path):
                self.app.logger.error(f"设备{self.device}SDK文件缺失: {sdk_dll_path}")
                return False

            self.app.logger.info(f"加载超脑设备{self.device}SDK: {sdk_dll_path}")
            self.hikSDK = load_library(sdk_dll_path)
            self.playM4SDK = load_library(play_dll_path)
            self.app.logger.info(f"超脑设备{self.device}SDK加载成功")
            return True
        except OSError as e:
            self.app.logger.error(f"超脑设备{self.device}SDK加载失败: {str(e)}")
            return False

    def _set_sdk_init_cfg(self):
        """初始化当前设备的SDK配置"""
        if sys.platform != "win32":
            self.app.logger.error(f"超脑设备{self.device}暂仅支持Windows系统")
            return False

        # 检查SDK路径
        if not os.path.exists(self.sdk_path):
            self.app.logger.error(f"超脑设备{self.device}SDK路径不存在: {self.sdk_path}")
            return False

        # 设置SDK路径
        sdk_path_cfg = NET_DVR_LOCAL_SDK_PATH()
        sdk_path_cfg.sPath = self.sdk_path.encode("gbk")
        if not self.hikSDK.NET_DVR_SetSDKInitCfg(
                NET_SDK_INIT_CFG_TYPE.NET_SDK_INIT_CFG_SDK_PATH.value, byref(sdk_path_cfg)
        ):
            self.app.logger.error(f"设备{self.device}设置SDK路径失败，错误码: {self.hikSDK.NET_DVR_GetLastError()}")
            return False

        # 设置OpenSSL库路径
        crypto_path = os.path.join(self.sdk_path, "libcrypto-1_1-x64.dll").encode("gbk")
        ssl_path = os.path.join(self.sdk_path, "libssl-1_1-x64.dll").encode("gbk")

        if not self.hikSDK.NET_DVR_SetSDKInitCfg(
                NET_SDK_INIT_CFG_TYPE.NET_SDK_INIT_CFG_LIBEAY_PATH.value, create_string_buffer(crypto_path)
        ):
            self.app.logger.error(f"设备{self.device}设置libcrypto失败，错误码: {self.hikSDK.NET_DVR_GetLastError()}")
            return False

        if not self.hikSDK.NET_DVR_SetSDKInitCfg(
                NET_SDK_INIT_CFG_TYPE.NET_SDK_INIT_CFG_SSLEAY_PATH.value, create_string_buffer(ssl_path)
        ):
            self.app.logger.error(f"设备{self.device}设置libssl失败，错误码: {self.hikSDK.NET_DVR_GetLastError()}")
            return False

        self.app.logger.info(f"超脑设备{self.device}SDK初始化配置完成")
        return True

    def init_sdk(self):
        """初始化SDK"""
        if self.hikSDK and self.playM4SDK:
            return True
        if not self._load_sdk():
            return False
        if not self._set_sdk_init_cfg():
            return False
        if not self.hikSDK.NET_DVR_Init():
            self.app.logger.error(f"设备{self.device}SDK核心初始化失败，错误码: {self.hikSDK.NET_DVR_GetLastError()}")
            return False
        # 设置日志路径
        log_path = bytes(self.log_path, encoding="utf-8")
        self.hikSDK.NET_DVR_SetLogToFile(3, log_path, False)
        return True

    # -------------------------- 设备登录与预览 --------------------------
    def login(self):
        """登录当前设备"""
        if self.user_id > -1:
            self.app.logger.info(f"设备{self.device}已登录，无需重复登录")
            return True

        # 使用当前设备的配置登录
        login_info = NET_DVR_USER_LOGIN_INFO()
        login_info.bUseAsynLogin = 0
        login_info.sDeviceAddress = self.ip.encode()
        login_info.wPort = self.port
        login_info.sUserName = self.username.encode()
        login_info.sPassword = self.password.encode()
        login_info.byLoginMode = 0

        self.device_info = NET_DVR_DEVICEINFO_V40()
        self.user_id = self.hikSDK.NET_DVR_Login_V40(byref(login_info), byref(self.device_info))

        if self.user_id < 0:
            self.app.logger.error(f"设备{self.device}登录失败，错误码: {self.hikSDK.NET_DVR_GetLastError()}")
            self.hikSDK.NET_DVR_Cleanup()
            return False

        # 计算总通道数
        analog_ch = self.device_info.struDeviceV30.byChanNum
        digital_ch = self.device_info.struDeviceV30.byIPChanNum + (self.device_info.struDeviceV30.byHighDChanNum << 8)
        self.total_channels = analog_ch + digital_ch
        serial_num = str(self.device_info.struDeviceV30.sSerialNumber, encoding="utf8").rstrip("\x00")
        self.app.logger.info(f"设备{self.device}登录成功！序列号: {serial_num}, 总通道数: {self.total_channels}")
        return True

    def _release_resource(self):
        """释放当前设备资源（增强版：确保所有句柄重置）"""
        self.app.logger.info(f"设备{self.device}开始释放资源...")
        # 1. 停止预览
        if self.real_play_handle >= 0:
            self.hikSDK.NET_DVR_StopRealPlay(self.real_play_handle)
            self.real_play_handle = -1
            self.app.logger.info(f"设备{self.device}已停止通道 {self.current_channel} 预览")
        # 2. 停止解码并释放播放端口
        if self.play_ctrl_port.value > -1:
            self.playM4SDK.PlayM4_Stop(self.play_ctrl_port)
            self.playM4SDK.PlayM4_CloseStream(self.play_ctrl_port)
            self.playM4SDK.PlayM4_FreePort(self.play_ctrl_port)
            self.play_ctrl_port = C_LONG(-1)
            self.app.logger.info(f"设备{self.device}已释放播放库端口和解码资源")
        # 3. 清空解码缓存
        with self.decode_lock:
            self.decode_buf = None
            self.frame_buffer.clear()
        self.app.logger.info(f"设备{self.device}资源释放完成")

    def switch_channel(self, channel_num: int):
        """切换当前设备的通道"""
        if not (1 <= channel_num <= self.total_channels):
            self.app.logger.error(f"设备{self.device}无效通道号: {channel_num}，有效范围1-{self.total_channels}")
            return False

        self._release_resource()
        self.current_channel = channel_num
        return self.start_preview()

    def start_preview(self):
        """启动当前设备的预览"""
        if self.user_id < 0:
            self.app.logger.error(f"设备{self.device}未登录，无法启动预览")
            return False

        # 获取播放库端口
        if not self.playM4SDK.PlayM4_GetPort(byref(self.play_ctrl_port)):
            self.app.logger.error(
                f"设备{self.device}获取播放端口失败，错误码: {self.playM4SDK.PlayM4_GetLastError(self.play_ctrl_port)}")
            return False

        # 预览参数：子码流（H.264）
        preview_info = NET_DVR_PREVIEWINFO()
        preview_info.hPlayWnd = 0  # 不绑定窗口
        preview_info.lChannel = self.current_channel
        preview_info.dwStreamType = 1  # 子码流
        preview_info.dwLinkMode = 0  # TCP模式
        preview_info.bBlocked = 1  # 阻塞取流
        preview_info.byProtoType = 0  # 私有协议

        # 启动预览（绑定裸流回调）
        self.real_play_handle = self.hikSDK.NET_DVR_RealPlay_V40(
            self.user_id, byref(preview_info), self.real_data_callback, None
        )

        if self.real_play_handle < 0:
            self.app.logger.error(f"设备{self.device}预览启动失败，错误码: {self.hikSDK.NET_DVR_GetLastError()}")
            self._release_resource()
            return False

        # 启用SDK解码
        if not self.playM4SDK.PlayM4_SetStreamOpenMode(self.play_ctrl_port, 0):
            self.app.logger.error(
                f"设备{self.device}设置流模式失败，错误码: {self.playM4SDK.PlayM4_GetLastError(self.play_ctrl_port)}")
            self._release_resource()
            return False

        if not self.playM4SDK.PlayM4_SetDecCallBackExMend(
                self.play_ctrl_port, self._decode_callback, None, 0, None
        ):
            self.app.logger.error(
                f"设备{self.device}注册解码回调失败，错误码: {self.playM4SDK.PlayM4_GetLastError(self.play_ctrl_port)}")
            self._release_resource()
            return False

        self.app.logger.info(f"设备{self.device}通道 {self.current_channel} 预览启动成功（已启用SDK解码）")
        return True

    # -------------------------- 裸流回调处理 --------------------------
    def _real_data_callback(self, l_play_handle, dw_data_type, p_buffer, dw_buf_size, p_user):
        """裸流回调：将数据交给PlayM4SDK解码"""
        with self.app.app_context():
            if dw_data_type == NET_DVR_SYSHEAD:
                self.app.logger.info(
                    f"设备{self.device}【系统头接收】通道{self.current_channel}，大小={dw_buf_size} bytes")
                # 系统头数据：打开流并初始化解码
                if not self.playM4SDK.PlayM4_OpenStream(self.play_ctrl_port, p_buffer, dw_buf_size, 1024 * 1024):
                    self.app.logger.error(
                        f"设备{self.device}打开流失败，错误码: {self.playM4SDK.PlayM4_GetLastError(self.play_ctrl_port)}")
                else:
                    if self.playM4SDK.PlayM4_Play(self.play_ctrl_port, 0):
                        self.app.logger.info(f"设备{self.device}【解码启动成功】通道{self.current_channel}，等待视频流...")
                    else:
                        self.app.logger.error(
                            f"设备{self.device}启动解码失败，错误码: {self.playM4SDK.PlayM4_GetLastError(self.play_ctrl_port)}")

            elif dw_data_type == NET_DVR_STREAMDATA:
                self.app.logger.debug(
                    f"设备{self.device}【视频流接收】通道{self.current_channel}，大小={dw_buf_size} bytes")
                # 视频流数据：输入到SDK解码
                if not self.playM4SDK.PlayM4_InputData(self.play_ctrl_port, p_buffer, dw_buf_size):
                    self.app.logger.error(
                        f"设备{self.device}输入流数据失败，错误码: {self.playM4SDK.PlayM4_GetLastError(self.play_ctrl_port)}")
                else:
                    self.app.logger.debug(
                        f"设备{self.device}【视频流接收】通道{self.current_channel}：成功输入{dw_buf_size} bytes")
            else:
                self.app.logger.debug(f"设备{self.device}接收非视频数据，类型: {dw_data_type}")

    # -------------------------- 帧获取与资源释放 --------------------------
    def get_latest_frame(self, timeout=0.1):
        """获取当前设备的最新帧"""
        with self.decode_lock:
            if self.decode_buf is not None and self.decode_buf.size > 0:
                frame = self.decode_buf.copy()
                self.frame_buffer.append(frame)
                return frame

            self.decode_lock.wait(timeout)

            if self.decode_buf is not None and self.decode_buf.size > 0:
                frame = self.decode_buf.copy()
                self.frame_buffer.append(frame)
                return frame
            elif len(self.frame_buffer) > 0:
                return self.frame_buffer[-1].copy()
            else:
                return None

    def logout(self):
        """登出当前设备（增强版：彻底释放所有资源）"""
        self._release_resource()
        if self.user_id > -1:
            self.hikSDK.NET_DVR_Logout(self.user_id)
            self.user_id = -1
            self.app.logger.info(f"设备{self.device}已登出")
        if self.hikSDK:
            self.hikSDK.NET_DVR_Cleanup()
            self.app.logger.info(f"设备{self.device}SDK资源已释放")
        # 重置所有状态（避免下次初始化时复用旧状态）
        self.decode_buf = None
        self.frame_buffer.clear()
        self.current_channel = 1
        self.total_channels = 0
        self.hikSDK = None
        self.playM4SDK = None

    def __del__(self):
        self.logout()