# -*- coding: utf-8 -*-
# @Version : Python3.8.20

from win32gui import (
    FindWindow,
    FindWindowEx,
    GetWindowText,
    GetWindowRect,
    GetForegroundWindow,
    SetForegroundWindow,
    SetWindowPos,
)
from win32process import GetWindowThreadProcessId
from win32com import client
from time import sleep


class GetHandle:
    def __init__(self) -> None:
        pass

    @staticmethod
    def find_window(class_name: str = None, window_name: str = None) -> int:
        """
        查找窗口
        :param class_name: 窗口类名
        :param window_name: 窗口标题
        :return: 窗口句柄
        """
        return FindWindow(class_name, window_name)

    @staticmethod
    def find_window_ex(
        parent_handle: int,
        child_after: int = None,
        class_name: str = None,
        window_name: str = None,
    ) -> int:
        """
        查找子窗口
        :param parent_handle: 父窗口句柄
        :param child_after: 子窗口句柄
        :param class_name: 窗口类名
        :param window_name: 窗口标题
        :return: 子窗口句柄
        """
        return FindWindowEx(parent_handle, child_after, class_name, window_name)

    @staticmethod
    def get_handle_title(handle: int) -> str:
        """
        获取窗口标题
        :param handle: 窗口句柄
        :return: 窗口标题
        """
        return GetWindowText(handle)

    @staticmethod
    def get_handle_pid(handle: int) -> int:
        """
        获取窗口进程ID
        :param handle: 窗口句柄
        :return: 窗口进程ID
        :raises ValueError: 如果获取进程ID失败
        """
        result = GetWindowThreadProcessId(handle)[1]
        if result >= 0:
            return result
        else:
            raise ValueError("获取窗口进程ID失败")

    @staticmethod
    def get_window_rect(handle: int = None) -> tuple:
        """
        获取窗口矩形
        :param handle: 窗口句柄
        :return: 窗口矩形
        """
        return GetWindowRect(handle)

    @staticmethod
    def handle_is_exist(self,handle: int) -> bool:
        """
        检查窗口句柄是否存在
        :param handle: 窗口句柄
        :return: True if the handle exists, False otherwise
        """
        if self.get_handle_title(handle) != "" or self.get_handle_pid(handle) is not None:
            return True
        else:
            return False

    @staticmethod
    def set_window_position(handle: int, pos: tuple, width: int =0, height: int = 0) -> bool:
        """
        设置窗口在屏幕上的位置
        :param handle: 窗口句柄
        :param pos: 窗口左上角的坐标 (x, y)
        :param width: 窗口宽度
        :param height: 窗口高度
        :return: True if the operation is successful, False otherwise
        """
        try:
            x1, y1, x2, y2 =  GetWindowRect(handle)
            if width == 0 and height == 0:
                width = x2 - x1
                height = y2 - y1
            SetWindowPos(handle, 0, pos[0], pos[1], width, height, 0)
            print(f"设置窗口位置和大小成功: {pos[0], pos[1], width, height}")
            return True
        except Exception as e:
            print(f"设置窗口位置和大小失败: {e}")
            return False

    @staticmethod
    def Click_mouse_get_handle(loop_times: int = 5) -> tuple:
        """
        等待用户点击窗口并获取窗口句柄和标题
        :param loop_times: 等待的循环次数（秒）
        :return: 窗体标题句柄和名称
        """
        hand_num = ""
        hand_win_title = ""
        for t in range(loop_times):
            print(f"<br>请在倒计时 [ {loop_times - t} ] 秒结束前，点击目标窗口")
            hand_num = GetForegroundWindow()
            hand_win_title = GetWindowText(hand_num)
            print(f"<br>目标窗口： [ {hand_win_title} ] [ {hand_num} ] ")
            sleep(1)  # 每1s输出一次
        x1, y1, x2, y2 = GetWindowRect(hand_num)
        print("<br>-----------------------------------------------------------")
        print(f"<br>目标窗口: [ {hand_win_title} ] 窗口大小：[ {x2 - x1} X {y2 - y1} ]")
        print("<br>-----------------------------------------------------------")
        return hand_win_title, hand_num
    @staticmethod
    def activate_window(handle: int) -> None:
        """
        激活窗口
        :param handle: 窗口句柄
        :return: None
        """
        try:
            SetForegroundWindow(handle)
        except Exception as e:
            print(f"激活窗口失败: {e}")


if __name__ == "__main__":
    hwnd = GetHandle.Click_mouse_get_handle(3)
    print(GetHandle.handle_is_exist(hwnd[1]))
    # activate_window(6555508)
