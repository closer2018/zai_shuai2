# -*- coding: utf-8 -*-
# @Version : Python3.8.20

from os.path import abspath, dirname
import cv2
import time
from subprocess import Popen, PIPE
import numpy as np
import win32com.client
from numpy import frombuffer, uint8, array
from win32con import SRCCOPY
from win32gui import (
    DeleteObject,
    SetForegroundWindow,
    GetWindowRect,
    GetWindowDC,
    ReleaseDC,
)
from win32ui import CreateDCFromHandle, CreateBitmap
from PIL import ImageGrab
from numpy import int32, float32
import dxcam
from ctypes import windll


class ImgProcess:
    """图像处理，传入的图片格式必须是cv2的格式"""

    def __init__(self):
        pass

    @staticmethod
    def save_img(img: np.ndarray, img_path_name: str = r"\images\screen_pic.jpg"):
        """保存内存中cv2格式的图片为本地文件"""
        if img is None:
            print("<br>未获取到需要保存的图片！")
        else:
            file_path = (
                abspath(dirname(__file__)) + img_path_name
            )  # 截图的存储位置，程序路径里面
            cv2.imwrite(
                file_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 20]
            )  # 保存截图 质量（0-100）

    @staticmethod
    def show_img(img: np.ndarray):
        """查看内存中cv2格式的图片"""
        if img is None:
            print("<br>未获取到需要显示的图片！")
        else:
            cv2.namedWindow("scr_img")  # 命名窗口
            cv2.imshow("scr_img", img)  # 显示
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    @staticmethod
    def draw_matching_region(img_src, template, top_left) -> np.ndarray:
        """
        在大图片中绘制匹配区域
        :param img_src: 需要绘制的大图图片
        :param template: 模板图片
        :param top_left: 匹配区域左上角坐标
        :return: 返回坐标(x,y) 与opencv坐标系对应
        """
        if top_left is None:
            print("<br>未获取坐标点位置！")
        else:
            h, w = template.shape[:2]
            bottom_right = (top_left[0] + w, top_left[1] + h)
            # 参数解释：图片，左上角坐标，右下角坐标，颜色，线宽
            img = cv2.rectangle(img_src, top_left, bottom_right, (0, 238, 118), 2)
            return img

    @staticmethod
    def img_compress(img: np.ndarray, compress_val: float = 0.5) -> np.ndarray:
        """压缩图片，默认0.5倍"""
        # height, width = img.shape[:2]  # 获取宽高
        height = img.shape[0]
        width = img.shape[1]

        # 压缩图片,压缩率compress_val
        size = (int(width * compress_val), int(height * compress_val))
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        return img

    @staticmethod
    def get_sift(img: np.ndarray):
        """
        :param img: 传入cv2格式的图片，获取特征点信息
        :return: 返回特征点信息
        """
        # 初始化SIFT探测器
        sift = cv2.SIFT_create()
        # cv.xfeatures2d.BEBLID_create(0.75)  # 已过时用法
        kp, des = sift.detectAndCompute(img, None)
        img_sift = [kp, des]
        return img_sift
    
    @staticmethod
    def is_corners_same_color(img_gray, img_color):
        """检测图片四个角颜色是否一致（灰度图判断）"""
        h, w = img_gray.shape[:2]
        
        # 获取四个角的像素值（灰度）
        corners = [
            img_gray[0, 0],          # 左上
            img_gray[0, w-1],        # 右上
            img_gray[h-1, 0],        # 左下
            img_gray[h-1, w-1]       # 右下
        ]
        
        # 获取原始彩色图像角点颜色（用于后续掩码生成）
        corner_color = img_color[0, 0]
        
        # 判断四个角灰度值是否相同
        return all(c == corners[0] for c in corners), corner_color


class GetScreenCapture:
    def __init__(self, handle: int = 0, handle_width: int = 0, handle_height: int = 0):
        super(GetScreenCapture, self).__init__()
        self.hwd_num = handle
        x1, y1, x2, y2 = GetWindowRect(handle)
        self.screen_width = x2 - x1
        self.screen_height = y2 - y1
        self.screen_scale_rate = (
            self.get_screen_scale_rate()
        )  # 尝试获取屏幕分辨率缩放比例（暂未能找到自动获取的方法）

    def window_screenshot(
        self, pos1: tuple = None, pos2: tuple = None, bg_capture_mode: int = 0
    ) -> np.ndarray:
        """
        windows窗体区域截图，可后台截图，可被遮挡，不兼容部分窗口
        Args:
        pos1 (tuple, optional): 截图区域左上角坐标 (x1, y1)。默认为None，表示截取整个窗口
        pos2 (tuple, optional): 截图区域右下角坐标 (x2, y2)。默认为None，表示截取整个窗口
        bg_capture_mode (int, optional): 截图模式，默认为0。
            - 0: PrintWindow模式,支持浏览器,速度慢
            - 1: BitBlt模式,速度快,不支持浏览器
        Returns: np.ndarray: 返回截图的numpy数组，格式为灰度图像
        Raises: ValueError: 当窗口句柄无效时抛出
        """
        # 根据缩放比例处理宽高
        hwnd = self.hwd_num
        width = self.screen_width
        height = self.screen_height
        # print("截图宽度：", width, "截图高度：", height)
        width = int(width / self.screen_scale_rate)  # 缩放
        height = int(height / self.screen_scale_rate)  # 缩放

        # 返回句柄窗口的设备环境，覆盖整个窗口，包括非客户区，标题栏，菜单，边框
        hwnd_dc = GetWindowDC(hwnd)
        # 创建设备描述表
        mfc_dc = CreateDCFromHandle(hwnd_dc)
        # 创建内存设备描述表
        save_dc = mfc_dc.CreateCompatibleDC()
        # 创建位图对象准备保存图片
        bit_map = CreateBitmap()
        # 为bitmap开辟存储空间
        bit_map.CreateCompatibleBitmap(mfc_dc, width, height)
        # 将截图保存到saveBitMap中
        save_dc.SelectObject(bit_map)

        try:
            if bg_capture_mode == 0:
                # PrintWindow模式
                windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 3)
            else:
                # BitBlt模式,保存bitmap到内存设备描述表
                save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), SRCCOPY)

            # 获取位图的字节数据
            bmp_str = bit_map.GetBitmapBits(True)
            # 将字节数据转换为 NumPy 数组
            image = frombuffer(bmp_str, dtype="uint8")
            image.shape = (
                height,
                width,
                4,
            )  # 重塑数组维度为3维图像数组，4: BGRA四个颜色通道(Blue,Green,Red,Alpha)
            # im_opencv = cv2.resize(im_opencv, (width, height))
            # print("<br>后台窗口截图成功！")

            # 根据坐标尺寸，重新调整图片截图位置
            if pos1 is not None and pos2 is not None:
                x1, y1 = pos1
                x2, y2 = pos2
                # 确保坐标在有效范围内
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(image.shape[1], int(x2))
                y2 = min(image.shape[0], int(y2))

                # 使用numpy切片操作裁剪指定区域
                # image[y1:y2, x1:x2] 表示:
                # y1:y2 - 裁剪高度范围(行)
                # x1:x2 - 裁剪宽度范围(列)
                image = image[y1:y2, x1:x2]

            return image
        finally:
            # 清理资源,内存释放
            DeleteObject(bit_map.GetHandle())
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            ReleaseDC(hwnd, hwnd_dc)

    def window_screenshot_bk(
        self, pos1: tuple = None, pos2: tuple = None, bk_capture_mode: int = 10
    ) -> np.ndarray:
        """
        使用PIL或者dxcam的方式进行窗口截图（前台方式）
        Args:
            bk_capture_mode: 10使用PIL截图，11使用dxcam截图
            pos1: 截图左上角坐标
            pos2: 截图右下角坐标
        Returns:
            np.ndarray: 灰度图像数据
        Raises:
            RuntimeError: 窗口操作失败时抛出
        """
        try:
            # shell = win32com.client.Dispatch("WScript.Shell")
            # shell.SendKeys("%")
            SetForegroundWindow(self.hwd_num)  # 窗口置顶

            # time.sleep(0.1)  # 置顶后等0.1秒再截图
            x1, y1, x2, y2 = GetWindowRect(self.hwd_num)  # 获取窗口坐标
            if pos1 is not None and pos2 is not None:
                x1, y1, x2, y2 = pos1[0], pos1[1], pos2[0], pos2[1]

            if None in (x1, y1, x2, y2):
                raise RuntimeError("获取窗口坐标失败")
            if bk_capture_mode == 10:
                grab_image = ImageGrab.grab((x1, y1, x2, y2))  # 用PIL方法截图
            elif bk_capture_mode == 11:
                camera = dxcam.create()
                grab_image = camera.grab((x1, y1, x2, y2))  # 用dxcam方法截图
            im_opencv = np.array(grab_image)  # 转换为cv2的矩阵格式
            # print("<br>前台截图成功！")
            return im_opencv

        except Exception as e:
            print(f"截图失败: {e}")
            return None

    @staticmethod
    def get_screen_scale_rate() -> float:
        """设置缩放比例"""
        my_screen_scale_rate = 1.0  # 缩放比例
        return float(my_screen_scale_rate)


class GetPosByTemplateMatch:
    def __init__(self):
        pass

    @staticmethod
    def get_pos_by_template(
        screen_capture: np.ndarray,
        target_pic: tuple,
        sim: float = 0.8,
        debug_status: bool = False,
        image_show: bool = False,
    ) -> tuple:
        """
        使用模板匹配在屏幕截图中查找目标图片
        模板匹配，速度快，但唯一的缺点是，改变目标窗体后，必须重新截取模板图片才能正确匹配
        Args:
            screen_capture: 屏幕截图数据
            target_pic: 要匹配的模板图片元组
            debug_status: 是否启用调试模式
            image_show: 是否显示匹配结果

        Returns:
            tuple: (匹配位置(x,y), 匹配的模板索引), 如果未找到则位置为 None,
            返回坐标(x,y) 与opencv坐标系对应, 以及与坐标相对应的图片在模板图片中的位置
        """
        try:
            screen_high = screen_capture.shape[0]
            screen_width = screen_capture.shape[1]

            # 获取目标点位置
            val= sim  # 设置相似度阈值

            for i, template in enumerate(target_pic):
                pos = GetPosByTemplateMatch.template_matching_igbg(
                    screen_capture,
                    template,
                    screen_width,
                    screen_high,
                    val,
                    debug_status,
                    image_show,
                    i,
                )
                if pos is not None:
                    return pos, i
            return None, -1
        except Exception as e:
            print(f"模板匹配失败: {e}")
            return None, -1

    @staticmethod
    def template_matching_igbg(
        img_src: np.ndarray,
        template: np.ndarray,
        screen_width: int,
        screen_height: int,
        val: float,
        debug_status: bool,
        image_show: bool,
        i: int,
    ) -> tuple:
        """
        使用模板匹配在源图像中查找目标模板
        Args:
            img_src: 大图，待检测的图像
            template: 模板图像
            screen_width: 屏幕宽度
            screen_height: 屏幕高度
            val: 匹配阈值
            debug_status: 是否开启调试模式
            i: 模板索引
        Returns:
            tuple: ((x, y), confidence) 匹配位置和置信度，未找到则返回 (None, 0.0)
        """
        if img_src is None or template is None:
            return None
        
        # 获取尺寸信息
        tem_h, tem_w = template.shape[:2]
        src_h, src_w = img_src.shape[:2]

        # 转换为灰度图
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

        # 检测四个角颜色是否一致
        same_corners, corner_color = ImgProcess.is_corners_same_color(template_gray, template)

        # 匹配参数初始化
        use_sqdiff = same_corners
        method = cv2.TM_SQDIFF_NORMED if use_sqdiff else cv2.TM_CCOEFF_NORMED

        # 创建掩码（仅SQDIFF需要）
        mask = None
        if use_sqdiff:
            mask = np.all(template != corner_color, axis=2).astype(np.uint8)

        # 执行模板匹配
        res = cv2.matchTemplate(target_gray, template_gray, method, mask=mask)

        # 结果解析
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = min_loc if use_sqdiff else max_loc
        confidence = (1 - min_val) if use_sqdiff else max_val  # 统一置信度表示


        if confidence >= val:  # 计算相对坐标
            # 坐标转换（考虑模板中心点）
            x_center = top_left[0] + tem_w // 2
            y_center = top_left[1] + tem_h // 2
            
            # 屏幕比例换算
            position = (
                int(screen_width * x_center / src_w),
                int(screen_height * y_center / src_h)
            )
            if debug_status:
                print(f"<br>第 [ {i+1} ] 张图片，匹配分数：[ {round(confidence,2)} ]")
            if image_show:  # 调试图片是否显示图片
                draw_img = ImgProcess.draw_matching_region(
                    img_src, template, top_left
                )
                ImgProcess.show_img(draw_img)
            return position
        else:
            if debug_status:
                print(f"<br>第 [ {i+1} ] 张图片，匹配分数：[ {round(confidence,2)} ]，分数低于{val}，不显示图片")
            return None

class GetPosBySiftMatch:
    def __init__(self):
        pass

    @staticmethod
    def get_pos_by_sift(
        target_sift, screen_sift, target_hw, target_img, screen_img, debug_status
    ):
        """
        特征点匹配，准确度不好说，用起来有点难受，不是那么准确（比如有两个按钮的情况下），但是待检测的目标图片不受缩放、旋转的影响
        :return: 返回坐标(x,y) 与opencv坐标系对应，以及与坐标相对应的图片在所有模板图片中的位置
        """
        # print("正在匹配…")
        pos = None
        i = 0
        for i in range(len(target_img)):
            # print(i)
            pos = GetPosBySiftMatch.sift_matching(
                target_sift[i],
                screen_sift,
                target_hw[i],
                target_img[i],
                screen_img,
                debug_status,
                i,
            )
            if pos is not None:
                break
        return pos, i

    @staticmethod
    def sift_matching(
        target_sift, screen_sift, target_hw, target_img, screen_img, debug_status, i
    ):
        """
        特征点匹配，准确度不好说，用起来有点难受，不是那么准确（比如有两个按钮的情况下），但是待检测的目标图片不受缩放、旋转的影响
        :param target_sift: 目标的特征点信息
        :param screen_sift: 截图的特征点信息
        :param target_hw: 目标的高和宽
        :param target_img: cv2格式的目标图片
        :param screen_img: cv2格式的原图
        :param debug_status: 调试模式
        :param i: 第几次匹配
        :return: 返回坐标(x,y) 与opencv坐标系对应
        """
        # 利用创建好的特征点检测器去检测两幅图像的特征关键点，
        # 其中kp含有角度、关键点坐标等多个信息，具体怎么提取出坐标点的坐标不清楚，
        # des是特征描述符，每一个特征点对应了一个特征描述符，由一维特征向量构成
        kp1 = target_sift[0]
        des1 = target_sift[1]
        kp2 = screen_sift[0]
        des2 = screen_sift[1]
        min_match_count = 9  # 匹配到的角点数量大于这个数值即匹配成功
        flann_index_kdtree = 0  # 设置Flann参数，这里是为了下一步匹配做准备
        index_params = dict(
            algorithm=flann_index_kdtree, trees=4
        )  # 指定匹配的算法和kd树的层数
        search_params = dict(checks=50)  # 指定返回的个数

        # 根据设置的参数创建特征匹配器
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # 利用创建好的特征匹配器利用k近邻算法来用模板的特征描述符去匹配图像的特征描述符，k指的是返回前k个最匹配的特征区域
        # 返回的是最匹配的两个特征点的信息，返回的类型是一个列表，列表元素的类型是Dmatch数据类型，具体是什么我也不知道
        matches = flann.knnMatch(des1, des2, k=2)

        # 设置好初始匹配值，用来存放特征点
        good = []
        for m, n in matches:
            """
            比较最近邻距离与次近邻距离的SIFT匹配方式：
            取一幅图像中的一个SIFT关键点，并找出其与另一幅图像中欧式距离最近的前两个关键点，在这两个关键点中，如果最近的距离除以次近的距离得到的比率ratio
            少于某个阈值T，则接受这一对匹配点。因为对于错误匹配，由于特征空间的高维性，相似的距离可能有大量其他的错误匹配，从而它的ratio值比较高。
            显然降低这个比例阈值T，SIFT匹配点数目会减少，但更加稳定，反之亦然。Lowe推荐ratio的阈值为0.8，但作者对大量任意存在尺度、旋转和亮度变化的两幅图片进行匹配，
            结果表明ratio取值在0. 4~0. 6 之间最佳，小于0.4的很少有匹配点，大于0. 6的则存在大量错误匹配点，所以建议ratio的取值原则如下:
            ratio=0. 4：对于准确度要求高的匹配；
            ratio=0. 6：对于匹配点数目要求比较多的匹配；
            ratio=0. 5：一般情况。
            """
            if (
                m.distance < 0.6 * n.distance
            ):  # m表示大图像上最匹配点的距离，n表示次匹配点的距离，若比值小于0.5则舍弃
                good.append(m)
        if debug_status:
            print(
                f"<br>第 [ {i+1} ] 张图片，匹配角点数量：[ {len(good)} ] ,目标数量：[ {min_match_count} ]"
            )
        if len(good) > min_match_count:
            src_pts = float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            m, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # 绘制匹配成功的连线
            if debug_status:
                if True:  # 调试图片是否显示图片
                    matches_mask = (
                        mask.ravel().tolist()
                    )  # ravel方法将数据降维处理，最后并转换成列表格式
                    draw_params = dict(
                        matchColor=(0, 255, 0),  # draw matches in green color
                        singlePointColor=None,
                        matchesMask=matches_mask,  # draw only inliers
                        flags=2,
                    )
                    img3 = cv2.drawMatches(
                        target_img, kp1, screen_img, kp2, good, None, **draw_params
                    )  # 生成cv2格式图片
                    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)  # 转RGB
                    ImgProcess.show_img(img3)  # 测试显示

            # 计算中心坐标
            h, w = target_hw
            pts = float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
                -1, 1, 2
            )
            if m is not None:
                dst = cv2.perspectiveTransform(pts, m)
                arr = int32(dst)
                pos_arr = arr[0] + (arr[2] - arr[0]) // 2
                pos = (int(pos_arr[0][0]), int(pos_arr[0][1]))
                return pos
            else:
                return None
        else:
            return None


class PicColor:
    def __init__(self, handle: int, capture_mode: int = 0):
        """
        找图模块，传入窗口句柄, 找图模式（分前台和后台）,前台屏幕截图模式
        Args:
            handle (int): 窗口句柄
            capture_mode (int, optional): 找图模式(分为后台00,01; 前台10,11)
                - 00: 后台PrintWindow模式,支持浏览器,速度慢,(默认)
                - 01: 后台BitBlt模式,速度快,不支持浏览器
                - 10：使用前台PIL截图(默认)
                - 11：使用前台dxcam截图
        """
        self.handle = handle
        self.capture_mode = capture_mode
        self.run = True
        if self.handle:
            x1, y1, x2, y2 = GetWindowRect(self.handle)
            self.window_pos1 = (x1, y1)
            self.window_pos2 = (x2, y2)
            # print("窗口句柄(", self.handle, ")的大小是：", x1, y1, x2, y2)
            self._window_h = y2 - y1
            self._window_w = x2 - x1

    def find_img(
        self,
        img_path_str: str,
        pos1: tuple = None,
        pos2: tuple = None,
        sim: float = 0.8,
        compress_img: float = 0,
        capture_mode: int = 0,
        debug_status: bool = False,
        image_show: bool = False,
    ) -> tuple:
        """
        在区域截图中查找目标图片
        Args:
            :param img_path_list: 图片路径字符串，用|分割多张图片路径
            :param pos1: 查找窗口范围左上角坐标 (x, y)
            :param pos2: 查找窗口范围右下角坐标 (x, y)
            :param sim: 相似度阈值，0-1之间,默认0.8
            :param compress_img: 压缩图片比例，0为不压缩
            :param capture_mode: 找图模式(分为后台00（默认），01和前台10，11)
                - 00: 后台PrintWindow模式,支持浏览器,速度慢
                - 01: 后台BitBlt模式,速度快,不支持浏览器
                - 10：使用前台PIL截图(默认)
                - 11：使用前台dxcam截图
            :param debug_status: 是否显示调试信息
            :param image_show: 是否显示匹配图片
        Returns:
            Tuple[Optional[Tuple[int, int]], int]:
                - 找到时返回((x, y), index)，index从1开始
                - 未找到时返回(None, -1)
        Raises:
            ValueError: 参数验证失败时抛出
        """
        # 参数验证
        try:
            if not img_path_str:
                raise ValueError("图片路径不能为空")
            if compress_img < 0 or compress_img > 1:
                raise ValueError("压缩比例必须在0-1之间")

            #  获取屏幕区域截图
            screen_img = GetScreenCapture(self.handle)
            if capture_mode != 00:  # 后台PrintWindow模式截图
                capture_mode = capture_mode
            else:
                capture_mode = self.capture_mode
            if capture_mode == 00:  # 后台PrintWindow模式截图
                screen_img = screen_img.window_screenshot(pos1, pos2, bg_capture_mode=0)
            elif capture_mode == 1:  # 后台BitBlt截图
                screen_img = screen_img.window_screenshot(pos1, pos2, bg_capture_mode=1)
            elif capture_mode == 10:  # 前台PIL截图
                screen_img = screen_img.window_screenshot_bk(
                    pos1, pos2, bk_capture_mode=10
                )
            elif capture_mode == 11:  # 前台dxcam截图
                screen_img = screen_img.window_screenshot_bk(
                    pos1, pos2, bk_capture_mode=11
                )

            if screen_img is None:
                raise RuntimeError("获取屏幕截图失败")
            if compress_img > 0:
                screen_img = ImgProcess.img_compress(
                    screen_img, compress_img
                )  # 压缩图片

            # 读取目标图像
            img_path_list = img_path_str.split("|")
            # print("图片路径列表：", img_path_list)
            target_img_list = []
            for img_path in img_path_list:
                # 读取目标图像
                # target_img = cv2.imread(img_path, 0)  # imread不能读取中文路径和名字图片
                target_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)

                if target_img is None:
                    print(f"<br>无法读取图像: {img_path}")
                    continue
                if compress_img > 0:
                    target_img = ImgProcess.img_compress(
                        target_img, compress_img
                    )  # 压缩图片
                target_img_list.append(target_img)  # 存到数组

            # 在截图中查找目标图像
            pos, i = GetPosByTemplateMatch.get_pos_by_template(
                screen_img, target_img_list,sim, debug_status, image_show
            )
            if pos:
                if compress_img > 0:
                    pos = [x / compress_img for x in pos]  # 还原缩放坐标
                if pos1 is not None and pos2 is not None:
                    pos = [pos[0] + pos1[0], pos[1] + pos1[1]]  # 还原区域截图坐标
                # 将坐标转换为整数
                pos = tuple(map(int, pos))
                if debug_status:
                    print(f"<br>在坐标 {pos} 找到第 {i+1} 张图像: {img_path_list[i]}")
                return pos, i + 1
            else:
                if debug_status:
                    print(f"<br>未找到图像: {img_path_list[i]}")
                return None, -1
        except Exception as e:
            if debug_status:
                print(f"查找图片失败: {e}")
            return None, -1

    def wait_find_img(
        self,
        img_path_str: str,
        pos1: tuple = None,
        pos2: tuple = None,
        wait_time: int = 10,
        sim: float = 0.8,
        compress_img: float = 0,
        capture_mode: int = 0,
        debug_status: bool = False,
        image_show: bool = False,
    ) -> tuple:
        """
        等待查找图片
        :param img_path_list: 图片路径，用|分割多张图片路径
        :param pos1: 查找范围左上角坐标
        :param pos2: 查找范围右下角坐标
        :param wait_time: 最大等待时间
        :param sim: 相似度阈值，0-1之间,默认0.8
        :param compress_img: 压缩图片比例，0为不压缩
        :param capture_mode: 找图模式(分为后台00（默认），01和前台10，11)
            - 00: 后台PrintWindow模式,支持浏览器,速度慢
            - 01: 后台BitBlt模式,速度快,不支持浏览器
            - 10：使用前台PIL截图(默认)
            - 11：使用前台dxcam截图
        :param debug_status: 是否显示调试信息
        :param image_show: 是否显示匹配图片
        :return: 返回找到的图像坐标,以及第几张图片(从1开始),如：[[x,y],1],失败返回None
        """
        start_time = time.time()
        while time.time() - start_time <= wait_time and self.run:
            pos, i = self.find_img(
                img_path_str,
                pos1,
                pos2,
                sim,
                compress_img,
                capture_mode,
                debug_status,
                image_show,
            )
            if pos is not None:
                pos = list(map(int, pos))
                return pos, i
            if wait_time <= 5:
                continue
                # time.sleep(0.1)
            elif wait_time <= 60:
                time.sleep(1)
            else:
                time.sleep(10)
        return None, -1


if __name__ == "__main__":
    hwnd = 986824
    pc = PicColor(hwnd, capture_mode=00)
    start_time = time.time()
    gs = GetScreenCapture(hwnd)


    # 查找图片
    # result = pc.find_img(
    #     r"images\金色1.bmp|images\金色.jpg",
    #     (10, 10),
    #     (500, 500),
    #     debug_status=True,
    #     image_show=True,
    # )

    result = pc.find_img(
        r"images\禅宗二级.bmp|images\魔族二级.bmp",
        sim = 0.98,
        debug_status=True,
        image_show=True,
    )
    print(result)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"程序执行时间: {execution_time} 秒")
