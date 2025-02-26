# -*- coding: utf-8 -*-
# @Version : Python3.8.20

import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from utils.GetHandle import GetHandle
from utils.PicColor import PicColor
from utils.MouseCtl import MouseCtl as Mouse
from utils.KeyboardCtl import KeyboardCtl as Keyboard
import time
from ctypes import windll
from threading import Thread
import keyboard
import pyautogui
from ctypes import wintypes

# 全局变量控制运行状态
running = True
need_restart = False





def hotkey_listener():
    """监听F12热键"""
    global running
    while True:
        if keyboard.is_pressed('F12'):
            running = False
            windll.user32.BlockInput(False)
            windll.user32.UnregisterHotKey(None, 1)
            print("F12被按下，已启用输入设备")
            sys.exit()
            break
        time.sleep(0.1)


def 进入副本(pc,mouse,key):
    global need_restart
    pos = pc.wait_find_img(r"images\秘境入口.bmp")
    if pos:
        mouse.left_click(pos[0])
        time.sleep(1)
    start_time = time.time()
    while time.time() - start_time < 60 : # 寻找秘境
        pos ,i  = pc.find_img(r"images\X级.bmp",sim = 0.95,debug_status = True)
        if pos:
            mouse.left_click(pos)
            mouse.left_click(pos)
            pos, i = pc.wait_find_img(r"images\是.bmp",wait_time = 3)
            if pos:
                time.sleep(0.2)
                key.key_press_bg("enter")
                time.sleep(2)
            break
        else:
            key.key_press_bg("page_down")
            time.sleep(0.7)
            if time.time() - start_time >= 30:
                print('秘境石头打完了')
                # 启用输入设备
                重启游戏(pc,mouse,key)
                need_restart = True
                time.sleep(1)
                break
    pos ,i  = pc.wait_find_img(r"images\魔族.bmp|images\禅宗.bmp|images\星空.bmp",sim = 0.95,wait_time = 3,debug_status = True)
    if i == 1 and need_restart == False :
        # 选择魔族
        print('魔族禁地')
        魔族秘境(pc,mouse,key)
    elif i == 2 and need_restart == False :
        # 选择禅宗
        print('禅宗遗址')
        禅宗秘境(pc,mouse,key)
    elif i == 3 and need_restart == False :
        # 选择星空
        print('星空古路')
        星空古路(pc,mouse,key)

def 魔族秘境(pc,mouse,key):
    点击小boss(pc,mouse,key)# 第一个小boss
    print('第一个小boss')
    战斗回合(pc,mouse,key)
    
    if need_restart ==False:
        for i in range(3):
            time.sleep(0.5)
            mouse.left_click((629,35))
            time.sleep(1.5)
        mouse.left_click((640,78))
        time.sleep(1)
        对话框(pc,mouse,key)
        print('第二个小boss')
        战斗回合(pc,mouse,key)
    
    if need_restart ==False:
        time.sleep(0.5)
        mouse.left_click((337,690))
        time.sleep(2)
        mouse.left_click((208,560))
        time.sleep(1)
        对话框(pc,mouse,key)
        print('第三个小boss')
        战斗回合(pc,mouse,key)

    if need_restart ==False:
        for i in range(2):
            time.sleep(0.5)
            mouse.left_click((1115,360))
            time.sleep(1.5)
        mouse.left_click((968,367))
        time.sleep(1)
        对话框(pc,mouse,key)
        print('第四个小boss')
        战斗回合(pc,mouse,key)
    
    if need_restart ==False:
        time.sleep(0.5)
        mouse.left_click((247,358))
        time.sleep(2)
        对话框(pc,mouse,key)
        print('大boss')
        战斗回合(pc,mouse,key)
        time.sleep(4)



def 禅宗秘境(pc,mouse,key):
    pos, i = pc.wait_find_img(r"images\小boss.bmp",wait_time = 3)
    if pos and need_restart ==False:
        mouse.left_click((160,223))
        time.sleep(2)
        对话框(pc,mouse,key)
        print('第一个小boss')
        战斗回合(pc,mouse,key)
    
    if need_restart ==False:
        time.sleep(0.5)
        mouse.left_click((1075,364))
        time.sleep(2)
        mouse.left_click((1116,319))
        time.sleep(1)
        对话框(pc,mouse,key)
        print('第二个小boss')
        战斗回合(pc,mouse,key)

    if need_restart ==False:
        time.sleep(0.5)
        mouse.left_click((673,29))
        time.sleep(1.5)
        mouse.left_click((680,271))
        time.sleep(1)
        对话框(pc,mouse,key)
        print('第三个小boss')
        战斗回合(pc,mouse,key)

    if need_restart ==False:
        time.sleep(0.5)
        mouse.left_click((198,408))
        time.sleep(2)
        mouse.left_click((165,270))
        time.sleep(1)
        对话框(pc,mouse,key)
        print('第四个小boss')
        战斗回合(pc,mouse,key)

    if need_restart ==False:
        time.sleep(0.5)
        mouse.left_click((1080,125))
        time.sleep(2)
        mouse.left_click((634,168))
        time.sleep(1)
        对话框(pc,mouse,key)
        print('大boss')
        战斗回合(pc,mouse,key)
        time.sleep(4)


def 星空古路(pc,mouse,key):
    pos, i = pc.wait_find_img(r"images\小boss.bmp",wait_time = 3)
    if pos and need_restart ==False:
        mouse.left_click((344,222))
        time.sleep(3)
        对话框(pc,mouse,key)
        print('第一个小boss')
        战斗回合(pc,mouse,key)
    
    if need_restart ==False:
        time.sleep(0.5)
        mouse.left_click((595,171))
        time.sleep(1)
        mouse.left_click((645,176))
        time.sleep(1)
        对话框(pc,mouse,key)
        print('第二个小boss')
        战斗回合(pc,mouse,key)

    if need_restart ==False:
        time.sleep(0.5)
        mouse.left_click((117,319))
        time.sleep(2)
        对话框(pc,mouse,key)
        print('第三个小boss')
        战斗回合(pc,mouse,key)

    if need_restart ==False:
        time.sleep(0.5)
        mouse.left_click((408,122))
        time.sleep(1.5)
        mouse.left_click((408,128))
        time.sleep(1.5)
        对话框(pc,mouse,key)
        print('第四个小boss')
        战斗回合(pc,mouse,key)

    if need_restart ==False:
        time.sleep(0.5)
        mouse.left_click((984,311))
        time.sleep(2)
        对话框(pc,mouse,key)
        print('大boss')
        战斗回合(pc,mouse,key)
        time.sleep(4)


        

def 点击小boss(pc,mouse,key):
    pos, i = pc.wait_find_img(r"images\小boss.bmp",wait_time = 2)
    mouse.left_click(pos)
    time.sleep(1)
    对话框(pc,mouse,key)

def 对话框(pc,mouse,key):
    pos , i = pc.wait_find_img(r"images\对话框.bmp",(627,697),(657,717),wait_time = 2)
    if pos:
        time.sleep(0.5)
        key.key_press_bg("enter")
        time.sleep(2)


def 战斗回合(pc,mouse,key):
    global need_restart
    start_time = time.time()
    pos, i = pc.wait_find_img(r"images\回合.bmp",(119,686),(174,715),wait_time = 3)
    num = 0
    if pos: # 进入战斗
        while time.time() - start_time <= (3*60): # 等3分钟
            pos, i = pc.find_img(r"images\回合.bmp",(119,686),(174,715))
            if pos:
                if num % 20 == 0:
                    print(num , '战斗回合')
                    mouse.move_to((278,397))
                    if num >= 360: 
                        windll.user32.BlockInput(False)
                        print('战斗时间过长，重启游戏')
                        num = 0
                        重启游戏(pc,mouse,key)
                        need_restart = True
                        time.sleep(1)
                        break
                key.key_down_bg("enter")
                time.sleep(0.5)
                num += 1
            elif pos == None:
                print('战斗结束')
                战斗结果(pc,mouse,key)
                break

def 重启游戏(pc,mouse,key):
    if pc.handle:
        key.key_press_bg("F5")
    
    time.sleep(3)
    intx = None
    inty = None
    intx, inty = pyautogui.locateCenterOnScreen(r"images\play.PNG", minSearchTime=(2*60))
    if intx and inty:
        print('检测到金色，进入读档界面')
        time.sleep(1)
        pyautogui.click(intx, inty)
    elif intx == None and inty == None:
        windll.user32.BlockInput(False) # 禁用输入
        sys.exit()
    parentHwnd = GetHandle.find_window(None, "再刷一把2")
    if parentHwnd: # 窗口移动到左上角
        GetHandle.set_window_position(parentHwnd, (0, 0))
        global hwnd
        hwnd = GetHandle.find_window_ex(parentHwnd, None, None, "Chrome Legacy Window")
        GetHandle.activate_window(hwnd)
        print('当前句柄：',hwnd)
        time.sleep(1)
        # 设置后台模式
        pc = PicColor(hwnd, capture_mode = 0)
        mouse = Mouse(hwnd, click_mod = 1)
        key = Keyboard(hwnd)
        pyautogui.click(100,100)
        time.sleep(1)
    
    for i in range(30):
        key.key_press_bg("enter")
        print('按回车')
        time.sleep(2)
        pos, i = pc.find_img(r"images\模式.bmp",(41,692),(81,712),sim = 0.95)
        if pos:
            print('检测到模式，进入游戏界面，重启完毕')
            break

    need_restart = False


def 战斗结果(pc,mouse,key):
    pos, i = pc.wait_find_img(r"images\esc.bmp",(797,666),(867,702),wait_time = 3)
    if pos:
        print('战斗胜利')
        # key.key_up_bg("enter")
        key.key_press_bg("esc")
        time.sleep(2)
        key.key_up_bg("enter")
        return True
    else:
        pos, i = pc.find_img(r"images\死亡.bmp")
        if pos:
            mouse.left_click(pos)
            time.sleep(18)
            for i in range(4):
                key.key_press_bg("enter")
                print('按回车')
                time.sleep(5)
            time.sleep(5)
        return False

def 副本结束(pc,mouse,key):
    pos, i = pc.wait_find_img(r"images\秘境入口.bmp",wait_time = 5)
    if pos:
        print('副本结束,打完保存')
        key.key_press_bg("`")
        time.sleep(5)
    else:
        print('没有出副本，未知错误, 重启游戏')
        重启游戏(pc,mouse,key)
        time.sleep(1)
        
def worker():
    try:
        # 启动热键监听线程
        Thread(target=hotkey_listener, daemon=True).start()
    
        parentHwnd = GetHandle.find_window(None, "再刷一把2")
        if parentHwnd:
            GetHandle.set_window_position(parentHwnd, (0, 0))
            global hwnd
            hwnd = GetHandle.find_window_ex(parentHwnd, None, None, "Chrome Legacy Window")
            print(hwnd)
        
        # 设置后台模式
        pc = PicColor(hwnd, capture_mode = 0)
        mouse = Mouse(hwnd, click_mod = 1)
        key = Keyboard(hwnd)

        time.sleep(5)
        windll.user32.BlockInput(True) # 禁用输入
        # 激活窗口
        GetHandle.activate_window(hwnd)
        # 允许F12的特殊处理,不被阻止
        hotkey_id = wintypes.INT()
        windll.user32.RegisterHotKey(None, 1, 0x4000, 0x7B)  # 0x7B是F12的键码
        while running:
            进入副本(pc,mouse,key)
            副本结束(pc,mouse,key)
            if True:
                pc = PicColor(hwnd, capture_mode = 0)
                mouse = Mouse(hwnd, click_mod = 1)
                key = Keyboard(hwnd)


    finally:
        windll.user32.BlockInput(False)
        windll.user32.UnregisterHotKey(None, 1)


if __name__ == '__main__':
    worker()


