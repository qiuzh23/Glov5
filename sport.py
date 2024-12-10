# glov5/sport.py
import cv2
import numpy as np
import torch
import serial
import time
import sys
from pathlib import Path
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
import detect

FILE = Path(__file__).resolve()  # 获取当前文件的绝对路径
ROOT = FILE.parents[0]           # 获取当前文件所在这一级的父目录

# 激光控制类
class LaserController:
    def __init__(self, port='/dev/ttyUSB0', baud=115200):
        self.port = port
        self.baud = baud
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.5)
        except serial.SerialException as e:
            print(f"无法打开串口: {e}")
            sys.exit(1)

    def send_command(self, commands):
        self.ser.write(bytes(commands))

    def initialize(self):
        self.send_command([0x24, 0x42, 0x30, 0x39, 0x35, 0x23])
        self.send_command([0x24, 0x43, 0x30, 0x39, 0x35, 0x23])
        self.send_command([0x24, 0x44, 0x30, 0x39, 0x30, 0x23])

    def on(self):
        self.send_command([0x24, 0x44, 0x31, 0x30, 0x30, 0x23])

    def off(self):
        self.send_command([0x24, 0x44, 0x30, 0x39, 0x30, 0x23])

    def close(self):
        if self.ser.is_open:
            self.ser.close()

# 配置参数
S_DISTANCE = 87000  # 临界面积
DELTA_S_DISTANCE = 50000  # 临界面积差
DELTA_X_DISTANCE = 100  # 临界偏转角度
MOVE_SPEED_FAST = 0.4  # 向前速度
MOVE_SPEED_SLOW = 0.2  # 微调速度
TURN_SPEED_FAST = 0.4  # 转向速度
TURN_SPEED_SLOW = 0.2  # 微调速度
ANGLE = 0.3  # 瞄准角度
SLEEP_T = 10  # 瞄准时间（秒）
FRAME_CENTER = (960, 540)  # 视觉中心 (1920//2, 1080//2)

def load_model(weights, device='cpu'):
    ckpt = torch.load(weights, map_location=device)
    model = ckpt['model'].to(device).float().eval()
    return model

def main():
    # 初始化频道
    ChannelFactoryInitialize(0)
    
    # 初始化视频客户端
    client = VideoClient()
    client.SetTimeout(3.0)
    client.Init()
    
    # 初始化机器人控制
    robot = SportClient()
    robot.Init()
    
    # 初始化激光控制
    laser = LaserController()
    laser.initialize()
    
    # 加载模型
    weights = '/home/unitree/unitree_sdk2_python/example/front_camera/glov5/balloon60.pt'
    device = "cuda:0"  # 根据需要修改
    model = load_model(weights, device)
    
    try:
        while True:
            window = detect.run_inference(model, detect.get_frame(), detect.IMG_SIZE, device, detect.CONF_THRESH, detect.IOU_THRESH, detect.MAX_DET)
            S = window[2] * window[3]
            delta_x = window[0] - FRAME_CENTER[0]
            delta_S = S_DISTANCE - S
    
            # 判断是否达到目标
            if abs(delta_S) <= DELTA_S_DISTANCE and abs(delta_x) <= DELTA_X_DISTANCE:
                robot.Move(0.0, 0.0, -0.1)
                time.sleep(1)
                for _ in range(10 * SLEEP_T):
                    robot.Euler(0.0, ANGLE, 0.0)  # 低头
                    laser.on()
                    time.sleep(0.1)
                continue
            else:
                laser.off()
    
            # 方向调整
            if delta_x > 4 * DELTA_X_DISTANCE:
                robot.Move(0.0, 0.0, -TURN_SPEED_FAST)
            elif delta_x < -4 * DELTA_X_DISTANCE:
                robot.Move(0.0, 0.0, TURN_SPEED_FAST)
            elif delta_x > DELTA_X_DISTANCE:
                robot.Move(0.0, 0.0, -TURN_SPEED_SLOW)
            elif delta_x < -DELTA_X_DISTANCE:
                robot.Move(0.0, 0.0, TURN_SPEED_SLOW)
    
            time.sleep(1)
    
            # 距离调整
            if S < S_DISTANCE / 4:
                robot.Move(MOVE_SPEED_FAST, 0.0, 0.0)  # 如果过远，走近
            elif S > S_DISTANCE * 2:
                robot.Move(-MOVE_SPEED_FAST, 0.0, 0.0)  # 如果过近，远离
            elif S < S_DISTANCE - DELTA_S_DISTANCE:
                robot.Move(MOVE_SPEED_SLOW, 0.0, 0.0)  # 微调走近
            elif S > S_DISTANCE + DELTA_S_DISTANCE:
                robot.Move(-MOVE_SPEED_SLOW, 0.0, 0.0)  # 微调远离
            time.sleep(1)
    except KeyboardInterrupt:
        print("终止程序")
    finally:
        laser.off()
        laser.close()
        robot.Release()

if __name__ == "__main__":
    main()
