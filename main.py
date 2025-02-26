# -*- coding: utf-8 -*-
# @Version : Python3.8.20

import ctypes

import os,sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import cv2
import numpy as np
from utils.GetHandle import GetHandle
from utils.PicColor import PicColor,ImgProcess
import time
from tasks.mi_jing import worker

def run_as_admin():
    """
    检查并以管理员身份运行当前脚本。
    """
    if not ctypes.windll.shell32.IsUserAnAdmin():
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, __file__, None, 1)
        sys.exit()

if __name__ == '__main__':
    run_as_admin()
        
    worker()