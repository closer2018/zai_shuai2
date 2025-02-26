# -*- coding: utf-8 -*-
# @Version : Python3.8.20

import math
import numpy as np
from win32gui import GetWindowRect, SetForegroundWindow
from win32api import MAKELONG, SendMessage
from win32con import (
    WM_LBUTTONUP,
    WM_LBUTTONDOWN,
    WM_ACTIVATE,
    WA_ACTIVE,
    WM_RBUTTONDOWN,
    WM_RBUTTONUP,
    WM_MOUSEWHEEL,
)
from ctypes import windll
from time import sleep
import random

PostMessageW = windll.user32.PostMessageW

WM_MOUSEMOVE = 0x0200
WM_RBUTTONDOWN = 0x0204
WM_RBUTTONUP = 0x0205
WM_MOUSEWHEEL = 0x020A


class MouseCtl:
    def __init__(self, handle: int, click_mod: int = 10) -> None:
        """
        初始化
            :param handle: 需要绑定的窗口句柄
        """
        self.handle = handle
        self.click_mod = ClickOffSet.create_click_mod(click_mod)
        if self.handle:
            x1, y1, x2, y2 = GetWindowRect(self.handle)
            self.window_pos1 = (x1, y1)
            self.window_pos2 = (x2, y2)
            # print("窗口句柄(",self.handle,")的大小是：",x1,y1,x2,y2)
            self._window_h = y2 - y1
            self._window_w = x2 - x1

    def move_to(self, pos: tuple) -> bool:
        """移动鼠标到坐标(x,y),可后台移动（仅兼容部分窗体程序）

        Args:
            pos (tuple): 坐标
        """
        if pos is not None:
            px, py = self.get_p_pos(
                self.click_mod, self._window_w, self._window_h, pos
            )  # 获取模型中的偏移坐标（九宫格）

            cx = int(px + pos[0])
            cy = int(py + pos[1])

            # 模拟鼠标指针移动
            wparam = 0
            lparam = MAKELONG(cx, cy)
            # SendMessage(self.handle, WM_ACTIVATE, WA_ACTIVE, 0)
            SendMessage(self.handle, WM_MOUSEMOVE, wparam, lparam)
            return True
        else:
            return False

    def left_down(self, pos: tuple):
        """在坐标(x,y)处左键按下,可后台点击（仅兼容部分窗体程序）

        Args:
            pos (tuple): 坐标
        """
        if pos is not None:
            px, py = self.get_p_pos(
                self.click_mod, self._window_w, self._window_h, pos
            )  # 获取模型中的偏移坐标（九宫格）

            cx = int(px + pos[0])
            cy = int(py + pos[1])

            # 模拟鼠标按下
            wparam = 0
            lparam = MAKELONG(cx, cy)
            # SendMessage(self.handle, WM_ACTIVATE, WA_ACTIVE, 0)
            SendMessage(self.handle, WM_LBUTTONDOWN, wparam, lparam)  # 模拟鼠标按下
            return lparam
        else:
            return False

    def left_up(self, lparam):
        """在坐标(x,y)处左键弹起,可后台点击（仅兼容部分窗体程序）

        Args:
            lparam: 鼠标按下时候的坐标，必须搭配left_down使用
        """
        # 模拟鼠标抬起
        wparam = 0
        sleep((random.randint(5, 15)) / 100)  # 点击弹起改为随机
        SendMessage(self.handle, WM_LBUTTONUP, wparam, lparam)  # 模拟鼠标弹起

    def left_click(self, pos: tuple) -> bool:
        """在坐标(x,y)处左键按下,可后台点击（仅兼容部分窗体程序）

        Args:
            pos (tuple): 坐标
        """
        if pos is not None:
            lparam = self.left_down(pos)
            self.left_up(lparam)
            return True
        else:
            return False

    def right_down(self, pos: tuple):
        """在坐标(x,y)处右键按下,可后台点击（仅兼容部分窗体程序）

        Args:
            pos (tuple): 坐标
        """
        if pos is not None:
            px, py = self.get_p_pos(
                self.click_mod, self._window_w, self._window_h, pos
            )  # 获取模型中的偏移坐标（九宫格）

            cx = int(px + pos[0])
            cy = int(py + pos[1])

            # 模拟鼠标按下
            wparam = 0
            lparam = MAKELONG(cx, cy)
            # SendMessage(self.handle, WM_ACTIVATE, WA_ACTIVE, 0)
            SendMessage(self.handle, WM_RBUTTONDOWN, wparam, lparam)  # 模拟鼠标右键按下
            return lparam
        else:
            return False

    def right_up(self, lparam):
        """在坐标(x,y)处右键弹起,可后台点击（仅兼容部分窗体程序）

        Args:
            lparam: 鼠标右键按下时候的坐标,必须搭配left_down使用
        """
        # 模拟鼠标抬起
        wparam = 0
        sleep((random.randint(50, 150)) / 1000)  # 点击弹起改为随机
        SendMessage(self.handle, WM_RBUTTONUP, wparam, lparam)  # 模拟鼠标右键弹起

    def right_click(self, pos: tuple) -> bool:
        """在坐标(x,y)处右键点击,可后台点击（仅兼容部分窗体程序）

        Args:
            pos (tuple): 坐标
        """
        if pos is not None:
            lparam = self.right_down(pos)
            self.right_up(lparam)
            return True
        else:
            return False

    def mouse_wheel(self, pos: tuple, delta: int) -> bool:
        """在坐标(x,y)处滚轮滚动,可后台点击（仅兼容部分窗体程序）

        Args:
            pos (tuple): 坐标
            delta (int): 滚动的距离，向上为正，向下为负，如向下滚动10个单位，则为-10
        """
        if pos is not None:
            self.move_to(pos)  # 移动到坐标
            delta = delta * 120
            wparam = delta << 16
            lparam = MAKELONG(pos[0], pos[1])
            # SendMessage(self.handle, WM_ACTIVATE, WA_ACTIVE, 0)
            SendMessage(self.handle, WM_MOUSEWHEEL, wparam, lparam)  # 模拟鼠标滚轮滚动
            return True
        else:
            return False

    def drag_bg(self, pos1: tuple, pos2: tuple) -> bool:
        """拖拽，功能未测试
           在坐标(x1,y1)处鼠标左键按下，在坐标(x2,y2)处鼠标左键弹起,可后台点击（仅兼容部分窗体程序）
        Args:
            pos1 (tuple): 坐标1
            pos2 (tuple): 坐标2
        """
        if pos1 is not None and pos2 is not None:
            move_x = np.linspace(pos1[0], pos2[0], num=20, endpoint=True)[0:]
            move_y = np.linspace(pos1[1], pos2[1], num=20, endpoint=True)[0:]
            # SendMessage(self.handle, WM_ACTIVATE, WA_ACTIVE, 0)
            SendMessage(self.handle, WM_LBUTTONDOWN, 0, MAKELONG(pos1[0], pos1[1]))
            for i in range(20):
                x = int(round(move_x[i]))
                y = int(round(move_y[i]))
                SendMessage(self.handle, WM_MOUSEMOVE, 0, MAKELONG(x, y))
                sleep(0.01)
            SendMessage(self.handle, WM_LBUTTONUP, 0, MAKELONG(pos2[0], pos2[1]))
            return True
        else:
            return False

    @staticmethod
    def get_p_pos(click_mod, width: int, height: int, pos: tuple) -> tuple:
        """获取模型中的偏移坐标（九宫格）"""

        # 以窗口中偏下（0.618）的位置为中心，旋转得到的坐标，抽取模型中的一个点并进行旋转
        # 得到的结果会根据原始坐标在窗口中的相对位置，形成一个与点击模型点击分布类似，但缩放方向不同的集合
        x1 = 0.382 * width
        x2 = 0.618 * width
        x3 = width
        y1 = 0.382 * height
        y2 = 0.618 * height
        y3 = height
        x = pos[0]
        y = pos[1]

        p_pos = ClickOffSet.choice_mod_pos(click_mod)

        if x <= x1 and y <= y1:
            # 左上
            px, py = ClickOffSet.pos_rotate(p_pos, 180)
        elif x <= x1 and y1 < y <= y2:
            # 左中
            px, py = ClickOffSet.pos_rotate(p_pos, 135)
        elif x <= x1 and y2 < y <= y3:
            # 左下
            px, py = ClickOffSet.pos_rotate(p_pos, 90)
        elif x1 < x <= x2 and y <= y1:
            # 中上
            px, py = ClickOffSet.pos_rotate(p_pos, 225)
        elif x1 < x <= x2 and y1 < y <= y2:
            # 中中，这个位置与左上一样处理
            px, py = ClickOffSet.pos_rotate(p_pos, 180)
        elif x1 < x <= x2 and y2 < y <= y3:
            # 中下
            px, py = ClickOffSet.pos_rotate(p_pos, 45)
        elif x2 < x <= x3 and y <= y1:
            # 右上
            px, py = ClickOffSet.pos_rotate(p_pos, 270)
        elif x2 < x <= x3 and y1 < y <= y2:
            # 右中
            px, py = ClickOffSet.pos_rotate(p_pos, 315)
        else:
            # 右下或其他情况
            px = p_pos[0]
            py = p_pos[1]

        py = int(py * 0.888)  # 让偏移结果再扁平一点

        return px, py


class ClickOffSet:
    def __init__(self):
        super(self).__init__()

    @staticmethod
    def create_click_mod(zoom, loc=0.0, scale=0.45, size=(2000, 2)):
        """
        生成正态分布的鼠标随机点击模型，zoom是缩放比例，约等于偏移像素点，size是模型大小即模型中的坐标总量
        """

        zoom = int(zoom)

        # 随机生成呈正态分布的聚合坐标（坐标0,0 附近概率最高）
        mx, my = zip(*np.random.normal(loc=loc, scale=scale, size=size))

        # 对原始数据进行处理，点击模型除正态分布外，参照人类的眼动模型行为，点击规律还应呈现一定的长尾效应，所以对第二象限进行放大，对第四象限缩小
        x_int = []
        y_int = []
        for t in range(len(mx)):

            # 对第二象限的坐标放大
            if mx[t] < 0 and my[t] > 0:
                x_int.append(int(mx[t] * zoom * 1.373))
                y_int.append(int(my[t] * zoom * 1.303))

            # 对第四象限的坐标缩小
            elif mx[t] > 0 and my[t] < 0:

                # 若第四象限全部缩小，会导致第四象限的密度偏大，所以把其中三分之一的坐标，转换为第二象限的坐标（第二象限放大后密度会变小）
                roll = np.random.randint(0, 9)
                if roll < 5:  # 转换其中二分之一的坐标
                    # pos = ClickModSet.pos_rotate([int(mx[t]), int(my[t])], 180)
                    # x_int.append(int(pos[0]))
                    # y_int.append(int(pos[1]))

                    x_int.append(int(mx[t] * zoom * -1.350))
                    y_int.append(int(my[t] * zoom * -1.200))

                    # x_int.append(int(mx[i] * zoom * -1))
                    # y_int.append(int(my[i] * zoom * -1))
                elif roll >= 8:  # 十分之二的坐标不处理
                    x_int.append(int(mx[t] * zoom))
                    y_int.append(int(my[t] * zoom))
                else:  # 剩下的坐标正常缩小
                    x_int.append(int(mx[t] * zoom * 0.618))
                    y_int.append(int(my[t] * zoom * 0.618))
            else:
                # 其他象限的坐标不变
                x_int.append(int(mx[t] * zoom))
                y_int.append(int(my[t] * zoom))

        # 处理边界问题，如果坐标点超出偏移范围，则缩小
        for t in range(len(x_int)):

            # # 先缩小，原始数据稍微超出了zoom的范围
            x_int[t] = int(x_int[t] * 0.816)
            y_int[t] = int(y_int[t] * 0.712)

            # 再判断是否超出边界，超出则再缩小超出的部分
            if abs(x_int[t]) > zoom:
                x_int[t] = int(x_int[t] * 0.718)
            if abs(y_int[t]) > zoom * 1.15:
                y_int[t] = int(y_int[t] * 0.618)

        # 合并数据
        mod_data = np.array(list(zip(x_int, y_int)))

        return mod_data

    @staticmethod
    def choice_mod_pos(data_list: list) -> tuple:
        """
        从模型中抽取一个坐标（x1,y1）
        """

        # 随机抽取（平均抽取，每个坐标抽取概率相同，多次抽取的样本约等于模型中的样本，结果数据也呈正态分布）
        roll = np.random.randint(0, len(data_list) - 1)
        x1 = data_list[roll][0]
        y1 = data_list[roll][1]

        # 在正态分布基础上，随机偏移，避免太过集中
        if abs(x1) <= 50 and abs(y1) <= 50:
            roll_seed = 5
        elif 50 < abs(x1) <= 100 and 50 < abs(y1) <= 100:
            roll_seed = 15
        elif abs(x1) <= 50 and 50 < abs(y1) <= 100:
            roll_seed = 10
        elif abs(y1) <= 50 and 50 < abs(x1) <= 100:
            roll_seed = 10
        else:
            roll_seed = 20

        # roll偏移量
        xp = np.random.randint(-roll_seed, roll_seed)
        yp = np.random.randint(-roll_seed, roll_seed)
        x1 = x1 + xp
        y1 = y1 + yp

        return x1, y1

    @staticmethod
    def pos_rotate(pos: tuple, r=90) -> tuple:
        """
        将一个坐标围绕原点（0，0）,进行顺时针旋转，默认90度
        """
        rx = pos[0]
        ry = pos[1]
        # 对每个坐标进行变换，参照数学公式变换，有一点点误差，但不影响
        ang = math.radians(r)  # 将角度转换成弧度
        new_x = int(rx * math.cos(ang) + ry * math.sin(ang))
        new_y = int(-rx * math.sin(ang) + ry * math.cos(ang))

        return new_x, new_y


from utils.GetHandle import GetHandle

if __name__ == "__main__":
    mc = MouseCtl(1837526)
    mc.left_click((592, 554))
    # mc.mouse_wheel((300, 300), -1)
    # GetHandle.activate_window(mc.handle)
    mc.drag_bg((280, 274), (390, 382))
    print(mc.handle)
