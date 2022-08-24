import math
import os
import sys
import threading
from datetime import datetime
import numpy as np
from PyQt5.QtCore import QUrl, QPoint, QRect
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QImage, QPen, QColor, QPixmap
from numpy import ma
import csv
from user_interface import Ui_MainWindow
from PyQt5.QtWidgets import QWidget, QApplication, QFileDialog, QLabel, QMainWindow, QTableWidgetItem
from moviepy.editor import *

import cv2
import queue
from collections import deque

from Detection.yolox.detect import YoloX
from Tracking.bytetrack import BYTETracker
from Detection.yolox.utils.visualize import _COLORS
#lấy icon hình ảnh của app khi khởi chạy
ROOT_DIR = os.path.abspath(os.curdir)
ICON_PATH = os.path.join(ROOT_DIR, 'static/icon.png')
##########
detector = YoloX()
tracker = BYTETracker(track_thresh=0.5, track_buffer=30,
                      match_thresh=0.8, min_box_area=10, frame_rate=30)
#khởi tạo yolox và bytetrack. match_thresh  là frame sau giống với frame trước 80% thì ghép chung 1 id

#########                       

q = queue.Queue()
pts = deque(maxlen=64)
qp = QtGui.QPainter()
boundaries = [[((0, 110, 95), (20, 255, 255))],  # Red
              [((20, 90, 60), (50, 200, 200))],  # Yellow
              [((80, 90, 110), (90, 150, 255))]]  # Green

#hệ hsv

global rect
global lane_left, lane_center, lane_right
global direct_left, direct_center, direct_right
global vertical

BLOW_THE_X_SIGN = 'green'#đèn nảo thi bắt
WRONG_LANE_X_SIGN = 'green'#đèn khi nào bắt sai làn


def getFrame(video_dir, myFrameNumber):#đầu vào là video và framenumber
    cap = cv2.VideoCapture(video_dir)#lấy tất cả các frame trong video
    cap.set(cv2.CAP_PROP_POS_FRAMES, myFrameNumber)#chỉ vào frame với số framenumber
    cap.grab()# lấy frame đó ra
    return cap.retrieve(0) #trả về frame đã lấy
    #lấy frame từ video


def QtImage(screen, img):#định nghĩa hình sẽ truyền vào giao diện(tất cả frame)
    img_height, img_width, img_colors = img.shape # shape trả về 3 giá trị là chiều cao, chiều rộng, và màu
    scale = screen / img_height #tính tỉ lệ lấy chiều cao của khung app chia cho chiều cao của hình sẽ ra tỉ lệ
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#định dạng lại màu về RGB
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)# định dạng lại size cho vừa khung giao diện
    height, width, bpc = img.shape#sau khi shape theo khung thì lấy shape lại lần nữa
    bytesPerLine = 3 * width#tính tổng số byte của hình
    return QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)


def Detect(det, frame):#hàm detect
    box_detects = det.detect(frame.copy())[0]#biến det là yolox. gọi vào vòng lặp từng frame một. xử lý frame được chọn detect vật thể  0 là box
    classes = det.detect(frame.copy())[1]#xác định loại vật thể(xe, người....) 1 là class
    confs = det.detect(frame.copy())[2]#độ tin cậy
    return np.array(box_detects).astype(int), np.array(confs), np.array(classes)
    #trả lại 2 tọa độ của hộp và lưu dưới dạng só nguyên


def Average(lst):
    return sum(lst) / len(lst)
    #hàm tính trung bình cộng của 1 list


def colorDetector(frame, _rect):
    num_rect = len(_rect)
    color_sum = [0, 0, 0]# chứa số lượng pixel của mỗi màu đỏ xanh vàng
    color = []
    for j in range(num_rect):#a b c d là tọa độ trên trái dưới phải của ô detect đèn
        a = rect[0][0].y()
        b = rect[0][0].x()
        c = rect[0][1].y()
        d = rect[0][1].x()
        traffic_light = frame[a:c, b:d]#crop cái khung vừa vẽ
        traffic_light = cv2.GaussianBlur(traffic_light, (5, 5), 0)#làm mượt(smoothing image) (khử nhiễu, loại bỏ các cạnh của hình)
        hsv = cv2.cvtColor(traffic_light, cv2.COLOR_BGR2HSV)#chuyển về hệ màu hsv
        for i in range(len(boundaries)):
            for (lower, upper) in boundaries[i]:
                mask = cv2.inRange(hsv, lower, upper)# inrange quét hình pixel trong ảnh hsv. nếu giá trị trong khoảng lower đến upper. nếu nằm trong khoảng sẽ trả về 1, còn không sẽ trả về 0
                color_sum[i] = ma.sum(mask)#tính tổng của mask. tổng những điểm ảnh nằm trong khoảng boudery. tổng nào lớn nhất thì giá trị là màu đó.
        color.append(color_sum.index(np.max(color_sum)))# lấy số thứ tự trong dict color_sum ở trên để xác định khung chứa màu gì
    color_lb = None# lấy trung bình giá trị số màu ở các ô detect vẽ. tính trung bình giá trị số thứ tự ở trên để xác định màu đèn 
    if len(color) > 0:
        average = Average(color)
        if average == 2:
            color_lb = 'green'
        elif average == 0:
            color_lb = 'red'
        else:
            color_lb = 'yellow'
    return color_lb


class CameraView(QWidget):
    def __init__(self, parent=None):
        super(CameraView, self).__init__(parent)
        self.image = None
        global rect
        global lane_left, lane_center, lane_right
        global direct_left, direct_center, direct_right

    def setImage(self, image):#đặt ảnh
        self.image = image #image của khung 
        img_size = image.size()#size của hình sau khi truyền vào khung
        self.setMinimumSize(img_size)#size hình nhỏ nhất là size của khung
        self.update()

    def paintEvent(self, event):# hành động vẽ
        qp.begin(self)#qp : qtpainter
        if self.image:
            qp.drawImage(QPoint(0, 0), self.image)
        br = QtGui.QBrush(QColor(0, 252, 156, 40))
        qp.setBrush(br)
        #set màu và độ đậm của nét bút

        for item in rect:
            pen = QPen(QColor(255, 255, 255), 2, QtCore.Qt.SolidLine)#loại đường 
            qp.setPen(pen)
            qp.drawRect(QRect(item[0], item[1]))
            #định nghĩa đường vẽ qua các pixel. vẽ hình chứ nhật

        for item in lane_left:
            pen = QtGui.QPen(QColor(255, 0, 0), 2, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.drawLine(item[0], item[1])#vẽ từ trên trái đến dưới phải item 0 là điểm trên trái.
            qp.drawText(item[0], str("line left 1"))
        for item in lane_center:
            pen = QtGui.QPen(QColor(255, 255, 0), 2, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.drawLine(item[0], item[1])
            qp.drawText(item[0], str("line center 1"))
        for item in lane_right:
            pen = QtGui.QPen(QColor(0, 0, 255), 2, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.drawLine(item[0], item[1])
            qp.drawText(item[0], str("line right 1"))

        for item in direct_left:
            pen = QtGui.QPen(QColor(255, 0, 0), 2, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.drawLine(item[0], item[1])
            qp.drawText(item[0], str("line left 2"))
        for item in direct_center:
            pen = QtGui.QPen(QColor(255, 255, 0), 2, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.drawLine(item[0], item[1])
            qp.drawText(item[0], str("line center 2"))
        for item in direct_right:
            pen = QtGui.QPen(QColor(0, 0, 255), 2, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.drawLine(item[0], item[1])
            qp.drawText(item[0], str("line right 2"))
        qp.end()


class ExportView(QWidget): #xuất hình ảnh
    def __init__(self, parent=None):
        super(ExportView, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        img_size = image.size()
        self.setMinimumSize(img_size)
        self.update()

    def paintEvent(self, event):
        qp.begin(self)
        if self.image:
            qp.drawImage(QPoint(0, 0), self.image)
        qp.end()


def ccw(A, B, C):#couter clockwise (ngược chiều kim đồng hồ) để kiểm tra 2 đường có giao nhau không( )
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A, B, C, D):#kết hợp chung với ccw- thuật toán có sẵn
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


class DrawObject(QWidget):#vẽ vật thể
    def __init__(self, parent=None):
        super(DrawObject, self).__init__(parent)
        self.image = None
        self.flag = None
        self.direct = None
        self.img_holder = None
        self.screen = None
        global rect
        global lane_left, lane_center, lane_right
        global direct_left, direct_center, direct_right
        rect = []#list để lưu tọa độ các hình chữ nhật
        lane_left, lane_center, lane_right = [], [], []
        direct_left, direct_center, direct_right = [], [], []
        self.begin = QPoint()#định nghĩa begin là 1 điểm có x , y là số nguyên
        self.end = QPoint()
        self.show()

    def setImage(self, image):
        self.image = image
        img_size = image.size()
        self.setMinimumSize(img_size)
        self.update()

    def setMode(self, mode_flag, direct_flag):
        self.flag = mode_flag#để gán vào checkbox
        self.direct = direct_flag#để gán hướng
        self.update()

    def goBack(self):
        if self.flag == 'rect' and len(rect) > 0:
            rect.pop()#đẩy giá trị gần nhất trong list ra màn hình
        elif self.flag == 'lane' and self.direct == 'left' and len(lane_left) > 0:
            lane_left.pop()
        elif self.flag == 'lane' and self.direct == 'center' and len(lane_center) > 0:
            lane_center.pop()
        elif self.flag == 'lane' and self.direct == 'right' and len(lane_right) > 0:
            lane_right.pop()
        elif self.flag == 'direct' and self.direct == 'left' and len(direct_left) > 0:
            direct_left.pop()
        elif self.flag == 'direct' and self.direct == 'center' and len(direct_center) > 0:
            direct_center.pop()
        elif self.flag == 'direct' and self.direct == 'right' and len(direct_right) > 0:
            direct_right.pop()
        self.update()

    def paintEvent(self, event):#sau khi loại bỏ 1 đường, hệ thống sẽ loại bỏ đường đó và vẽ lại từ đầu để chúng ta không nhận ra
        qp.begin(self)
        if self.image:
            qp.drawImage(QPoint(0, 0), self.image)
        br = QtGui.QBrush(QColor(0, 252, 156, 40))
        qp.setBrush(br)

        for item in rect:
            pen = QtGui.QPen(QColor(255, 255, 255), 2, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.drawRect(QRect(item[0], item[1]))

        for item in lane_left:
            pen = QtGui.QPen(QColor(255, 0, 0), 2, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.drawLine(item[0], item[1])
            qp.drawText(item[0], str("line left 1"))
        for item in lane_center:
            pen = QtGui.QPen(QColor(255, 255, 0), 2, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.drawLine(item[0], item[1])
            qp.drawText(item[0], str("line center 1"))
        for item in lane_right:
            pen = QtGui.QPen(QColor(0, 0, 255), 2, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.drawLine(item[0], item[1])
            qp.drawText(item[0], str("line right 1"))

        for item in direct_left:
            pen = QtGui.QPen(QColor(255, 0, 0), 2, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.drawLine(item[0], item[1])
            qp.drawText(item[0], str("line left 2"))
        for item in direct_center:
            pen = QtGui.QPen(QColor(255, 255, 0), 2, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.drawLine(item[0], item[1])
            qp.drawText(item[0], str("center line 2"))
        for item in direct_right:
            pen = QtGui.QPen(QColor(0, 0, 255), 2, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.drawLine(item[0], item[1])
            qp.drawText(item[0], str("right line 2"))
        qp.end()

    def mousePressEvent(self, event):#click chuột
        self.begin = event.pos()#vị trí ban đầu
        self.end = event.pos()#vị trí cuối cùng. #đoạn dưới dùng để block người dùng chỉ được vẽ 1 đường mỗi làn. nếu đã tồn tại 1 đường trước đó thì nó sẽ xóa đi
        if self.direct == 'left':
            if self.flag == 'lane' and len(lane_left) == 1:
                lane_left.pop()
            elif self.flag == 'direct' and len(direct_left) == 1:
                direct_left.pop()
        elif self.direct == 'center':
            if self.flag == 'lane' and len(lane_center) == 1:
                lane_center.pop()
            elif self.flag == 'direct' and len(direct_center) == 1:
                direct_center.pop()
        elif self.direct == 'right':
            if self.flag == 'lane' and len(lane_right) == 1:
                lane_right.pop()
            elif self.flag == 'direct' and len(direct_right) == 1:
                direct_right.pop()
        self.update()

    def mouseMoveEvent(self, event):
        self.end = event.pos()
        self.update()
        #di chuyển chuột đến đâu thì sẽ thêm biến end đến đó

    def mouseReleaseEvent(self, event):#nhấc chuột lên lưu vị trí vào end. ghì đè vào giá trị end ở trên
        self.end = event.pos()
        self.update()
        if self.direct == 'left':
            if self.flag == 'lane':
                lane_left.append([self.begin, self.end])
            elif self.flag == 'direct':
                direct_left.append([self.begin, self.end])
        elif self.direct == 'center':
            if self.flag == 'lane':
                lane_center.append([self.begin, self.end])
            elif self.flag == 'direct':
                direct_center.append([self.begin, self.end])
        elif self.direct == 'right':
            if self.flag == 'lane':
                lane_right.append([self.begin, self.end])
            elif self.flag == 'direct':
                direct_right.append([self.begin, self.end])
        elif self.flag == 'rect':
            rect.append([self.begin, self.end])


def midPoint(a, b, c, d):
    return int(a + (c - a) / 2), int(b + (d - b) / 2)
    #tìm điểm giữa của boudingbox. vì giá trị trả về là trên trái dưới phải


def shortDir(file_dir):#lấy file
    stringNum = len(file_dir)
    if stringNum > 35:
        file_dir = "..." + file_dir[-25:]#độ dài đường dẫn file >35 thì thay bằng dấu ...
    return file_dir


class MyWindow(QMainWindow):
    def __init__(self, parent=None):
        super(QMainWindow, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowIcon(QtGui.QIcon(ICON_PATH))
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.openFileNameLabel = QLabel()
        self.saveFolderLabel = QLabel()
        self.fileDir = None
        self.saveDir = None
        self.ui.Browser.clicked.connect(self.setOpenFileName)
        self.ui.Browser_2.clicked.connect(self.setSaveFolder)
        self.ui.Play.clicked.connect(self.startVideo)
        self.ui.Stop.clicked.connect(self.stopVideo)
        self.ui.Line_left.clicked.connect(self.setLeftLane)
        self.ui.Line_Center.clicked.connect(self.setCenterLane)
        self.ui.Line_Right.clicked.connect(self.setRightLane)
        self.ui.Line_left_2.clicked.connect(self.setLeftDirect)
        self.ui.Line_Center_2.clicked.connect(self.setCenterDirect)
        self.ui.Line_Right_2.clicked.connect(self.setRightDirect)
        self.ui.Square.clicked.connect(self.setRect)
        self.ui.Delete.clicked.connect(self.GoBack)
        self.ui.Save_video.clicked.connect(self.exportVideo)
        self.ui.Table.itemDoubleClicked.connect(self.openImage)
        self.ui.Table.setColumnWidth(0, 300)
        self.ui.Table.setColumnWidth(1, 100)
        self.ui.Close.clicked.connect(self.close)
        self.ui.Minimun.clicked.connect(self.showMinimized)
        self.capture_thread = None
        self.saving_thread = None
        self.stop = True
        self.begin = QPoint()
        self.end = QPoint()
        self.previous = {}
        self.current = {}
        self.counter = {}
        self.mapping = {}
        self.violation = []#tên loại vi phạm
        self.k_counter = None
        self.t_counter1 = []
        self.t_counter2 = []
        self.t_counter3 = []
        self.t_counter4 = []
        self.t_counter5 = []
        self.t_counter6 = []
        self.v_counter = []#biến đếm vi phạm khi vượt đèn đỏ
        self.w_counter = []#biến đếm vi phạm sai làn
        self.current_time = 0
        self.screen = self.ui.Camera_view.frameSize().height()
        self.ui.Camera_view = CameraView(self.ui.Camera_view)
        self.ui.Draw_line = DrawObject(self.ui.Draw_line)

    def setOpenFileName(self):
        self.fileDir, _ = QFileDialog.getOpenFileName(
            self,
            "Open video", self.openFileNameLabel.text(),
            "Videos (*.mp4)"
        )
        if self.fileDir:
            fileName = shortDir(self.fileDir)
            self.ui.Filename.setText(fileName)
            retval, img = getFrame(self.fileDir, 5)
            qImg = QtImage(self.screen, img)
            self.ui.Draw_line.setImage(qImg)
            self.saveDir = os.path.dirname(self.fileDir)
            self.ui.Filename_2.setText(self.saveDir)
            self.showSaveDir()

    def setSaveFolder(self):
        self.saveDir = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if self.saveDir:
            self.showSaveDir()

    def showSaveDir(self):
        folderName = self.saveDir
        folderName = shortDir(folderName)
        self.ui.Filename_2.setText(folderName)
        Saving_info = "Images will be saved as: <font color='green'> {} </font> <br>" \
                      "Videos will be saved as: <font color='green'> {} </font>".format(folderName + "/Images",
                                                                                        folderName + "/Videos")
        self.ui.Result_dir.setText(Saving_info)

    def setLeftLane(self):
        self.ui.Draw_line.setMode("lane", "left")

    def setCenterLane(self):
        self.ui.Draw_line.setMode("lane", "center")

    def setRightLane(self):
        self.ui.Draw_line.setMode("lane", "right")

    def setLeftDirect(self):
        self.ui.Draw_line.setMode("direct", "left")

    def setCenterDirect(self):
        self.ui.Draw_line.setMode("direct", "center")

    def setRightDirect(self):
        self.ui.Draw_line.setMode("direct", "right")

    def setRect(self):
        self.ui.Draw_line.setMode("rect", None)

    def GoBack(self):
        self.ui.Draw_line.goBack()

    def startVideo(self):
        if self.fileDir:
            self.stop = False
            images_path = os.path.join(self.saveDir, "Images")
            videos_path = os.path.join(self.saveDir, "Videos")
            if not os.path.exists(images_path):
                os.makedirs(images_path)
            if not os.path.exists(videos_path):
                os.makedirs(videos_path)
            self.capture_thread = threading.Thread(target=self.update)
            self.capture_thread.start()

    def stopVideo(self):
        self.stop = True

    def openImage(self, index):#ấn vào tên ảnh hiện ra ảnh trong tab export
        keys = list(self.mapping.keys())
        file = 'Images/{}'.format(keys[index.row()])
        file_dir = os.path.join(self.saveDir, file)
        pixmap = QPixmap(file_dir)
        pixmap = pixmap.scaledToWidth(551)
        self.ui.Preview.setPixmap(pixmap)
        self.ui.Preview_name.setText(file)
        self.ui.Preview.show()

    def exportVideo(self):#xuất video khi xem ảnh
        self.saving_thread = threading.Thread(target=self.savingVideo)
        self.saving_thread.start()

    def savingVideo(self):
        index = (self.ui.Table.selectionModel().currentIndex())
        keys = list(self.mapping.keys())
        file = 'Videos/{}.mp4'.format(keys[index.row()])
        file_dir = os.path.join(self.saveDir, file)
        clip = VideoFileClip(self.fileDir)
        start, end = self.mapping[keys[index.row()]]
        clip = clip.subclip(start, end)
        clip.write_videofile(file_dir, audio=False, verbose=False, logger=None)

    def updateTable(self, file_name):
        if self.current_time - 2 > 0:#current_time là thời gian vi phạm
            start = self.current_time - 2
        else:
            start = 0
        end = self.current_time + 2
        self.mapping[file_name] = [start, end]
        num_cols = 1
        num_rows = len(self.mapping.keys())
        self.ui.Table.setColumnCount(num_cols)
        self.ui.Table.setRowCount(num_rows)
        idx = 0
        for key, value in self.mapping.items():
            self.ui.Table.setItem(idx, 0, QTableWidgetItem('{}.jpg'.format(key)))
            idx += 1

    def updateWrongLane(self, img, x0, y0, violation, track_id, time_stamp):
        self.w_counter.append(track_id)#thêm vào list vi phạm
        self.ui.Vcounter_2.setText(str(len(list(set(self.w_counter)))))
        cv2.rectangle(img, (x0, y0 - 10), (x0 + 10, y0), (255, 0, 0), -1)
        file_name = "{}_{}".format(violation, time_stamp)
        if self.ui.SaveImage.isChecked():#check tại checkbox setting ban đầu.
            cv2.imwrite("{}/Images/{}.jpg"
                        .format(self.saveDir, file_name), img)
            self.ui.Violation_name.setText(file_name)
            if self.ui.SaveVideo.isChecked():
                self.updateTable(file_name)

    def updateCrossLight(self, img, x0, y0, track_id, time_stamp):
        self.v_counter.append(track_id)
        self.ui.Vcounter.setText(str(len(list(set(self.v_counter)))))
        cv2.rectangle(img, (x0, y0 - 10), (x0 + 10, y0), (0, 0, 255), -1)
        file_name = "vuot_den_{}".format(time_stamp)
        if self.ui.SaveImage.isChecked():
            cv2.imwrite("{}/Images/{}.jpg"
                        .format(self.saveDir, file_name), img)
            self.ui.Violation_name.setText(file_name)
            if self.ui.SaveVideo.isChecked():
                self.updateTable(file_name)


    def update(self):
        global vertical
        cap = cv2.VideoCapture(self.fileDir)#cap videp
        while cap.isOpened():#khi đang chạy video
            if self.stop is True or not cap.isOpened():
                self.stop = True
                break
            ret, img = cap.read()#ret để giữ lại giá trị trả về của các frame, img dùng để lấy các frame
            img_height, img_width, img_colors = img.shape
            scale = self.screen / img_height
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            current_color = colorDetector(img, rect)#màu của hình vẽ
            current_frame = cap.get(1)#frame hiện tại 
            fps = cap.get(5)#fps hiện tại
            self.current_time = int(current_frame / fps)#tính thời gian vi phạm
            time_stamp = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
            box_detects, scores, classes = Detect(detector, img)#detector là yolox, img là frame hiện tại
            ind = [i for i, item in enumerate(classes) if item == 0]
            classes = np.delete(classes, ind)
            box_detects = np.delete(box_detects, ind, axis=0)
            scores = np.delete(scores, ind)
            data_track = tracker.update(box_detects, scores, classes)#dùng bytetrack để update id
            labels = detector.names

            for i in range(len(data_track)):#chạy tất cả track trong 1 frame
                box = data_track[i][:4]#track từ 0 đến 3 (tọa độ của hộp) vị trí của hộp, 2 điểm trên trái dưới phải
                track_id = int(data_track[i][4]) #thông số thứ 4 trong list đại diện cho id
                cls_id = int(data_track[i][5])#số thứ 5 trong list là đại diện cho class( người, xe máy, ô tô...)
                x0 = int(box[0])
                y0 = int(box[1])
                x1 = int(box[2])
                y1 = int(box[3])
                self.current[track_id] = midPoint(x0, y0, x1, y1)#biết điểm trước ddiemr sau để xác định có đi qua vạch hay không
                color = (_COLORS[track_id % 30] * 255).astype(np.uint8).tolist()#phân biệt các class với nhau(các hộp màu khác nhau cho các class)
                text = labels[cls_id] + "_" + str(track_id)#theo mã màu của color trong class
                txt_color = (0, 0, 0) if np.mean(_COLORS[track_id % 30]) > 0.5 else (255, 255, 255)
                font = cv2.FONT_HERSHEY_SIMPLEX
                txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
                txt_bk_color = (_COLORS[track_id % 30] * 255 * 0.7).astype(np.uint8).tolist()
                if self.ui.ShowLabel.isChecked():#show box trong settting
                    cv2.rectangle(img, (x0, y0 + 1), (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])), txt_bk_color, -1)
                    cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
                if self.ui.ShowBox.isChecked():
                    cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
                if track_id in self.previous:#nếu track id đã xuất hiện trong các frame trước thì sẽ chạy vòng ở dưới
                    cv2.line(img, self.previous[track_id], self.current[track_id], color, 1)# nối vị trí vật thể ở frame trước và frame sau
                    line_group0 = [lane_left, lane_center, lane_right]
                    for element in line_group0:
                        if len(element) > 0:
                            start_line = element[0][0].x(), element[0][0].y()
                            end_line = element[0][1].x(), element[0][1].y()
                            if intersect(self.previous[track_id], self.current[track_id], start_line, end_line):#nếu vật thể vượt qua 1 đường
                                if line_group0.index(element) == 0:#số thự tự trong list =0 ( đường bên trái)
                                    self.t_counter1.append(track_id)#cập nhật id vật thể vào danh sách các vật thể đi qua đường số 0
                                    if current_color == BLOW_THE_X_SIGN and not self.ui.red_light_left.isChecked():
                                        self.updateCrossLight(img, x0, y0, track_id, time_stamp)#tính vi phạm vượt đèn
                                elif line_group0.index(element) == 1:
                                    self.t_counter2.append(track_id)
                                    if current_color == BLOW_THE_X_SIGN and not self.ui.red_light_center.isChecked():
                                        self.updateCrossLight(img, x0, y0, track_id, time_stamp)
                                elif line_group0.index(element) == 2:
                                    self.t_counter3.append(track_id)
                                    if current_color == BLOW_THE_X_SIGN and not self.ui.red_light_right.isChecked():
                                        self.updateCrossLight(img, x0, y0, track_id, time_stamp)

                    '''
                    t_counter1: line 1
                    t_counter2: line 2
                    t_counter3: line 3
                    t_counter4: line 4
                    t_counter5: line 5
                    t_counter6: line 6
                    '''
                    line_group1 = [direct_left, direct_center, direct_right]
                    for element in line_group1:
                        if len(element) > 0:
                            start_line = element[0][0].x(), element[0][0].y()
                            end_line = element[0][1].x(), element[0][1].y()
                            if intersect(self.previous[track_id], self.current[track_id], start_line, end_line):

                                '''
                                xét tại thời điểm cắt qua line 4
                                '''
                                if line_group1.index(element) == 0:
                                    self.t_counter4.append(track_id)#tại thời điểm vượt trái
                                    self.ui.Tcounter_3.setText(str(len(list(set(self.t_counter4)))))#thêm vào danh sách các vật đã vượt qua đường 4
                                    if current_color == WRONG_LANE_X_SIGN:
                                        if track_id in self.t_counter2 and \
                                                not self.ui.turn_left_center_lane.isChecked():
                                            violation = "sai_lan_giua"
                                            self.updateWrongLane(img, x0, y0, violation, track_id, time_stamp)
                                        #nếu track id đã vượt qua đường 2 và ô checkbox đường giữa được rẽ trái k được tích thì nó vi phạm
                                        elif track_id in self.t_counter3 and \
                                                not self.ui.turn_left_right_lane.isChecked():
                                            violation = "sai_lan_phai"
                                            self.updateWrongLane(img, x0, y0, violation, track_id, time_stamp)

                                '''
                                xét tại thời điểm cắt qua line 5
                                '''
                                if line_group1.index(element) == 1:
                                    self.t_counter5.append(track_id)
                                    self.ui.Tcounter_4.setText(str(len(list(set(self.t_counter5)))))
                                    if current_color == WRONG_LANE_X_SIGN:
                                        if track_id in self.t_counter1 and \
                                                not self.ui.go_forward_left_lane.isChecked():
                                            violation = "sai_lan_trai"
                                            self.updateWrongLane(img, x0, y0, violation, track_id, time_stamp)
                                        elif track_id in self.t_counter3 and \
                                                not self.ui.go_forward_right_lane.isChecked():
                                            violation = "sai_lan_phai"
                                            self.updateWrongLane(img, x0, y0, violation, track_id, time_stamp)

                                '''
                                xét tại thời điểm cắt qua line 6
                                '''
                                if line_group1.index(element) == 2:
                                    self.t_counter6.append(track_id)
                                    self.ui.Tcounter_5.setText(str(len(list(set(self.t_counter6)))))
                                    if current_color == WRONG_LANE_X_SIGN:
                                        if track_id in self.t_counter1 and \
                                                not self.ui.turn_right_left_lane.isChecked():
                                            violation = "sai_lan_trai"
                                            self.updateWrongLane(img, x0, y0, violation, track_id, time_stamp)
                                        elif track_id in self.t_counter2 and \
                                                not self.ui.turn_right_center_lane.isChecked():
                                            violation = "sai_lan_giua"
                                            self.updateWrongLane(img, x0, y0, violation, track_id, time_stamp)

                    self.ui.Tcounter_0.setText(str(len(list(set(self.t_counter1)))))#lưu số lượng crossing ở setting
                    self.ui.Tcounter_1.setText(str(len(list(set(self.t_counter2)))))
                    self.ui.Tcounter_2.setText(str(len(list(set(self.t_counter3)))))
                self.previous[track_id] = self.current[track_id]
            wd01 = QtImage(self.screen, img)
            self.ui.Camera_view.setImage(wd01)

    def Checking(self):
        cap = cv2.VideoCapture(self.fileDir)
        myFrameNumber = 5
        totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if myFrameNumber >= 0 & myFrameNumber <= totalFrames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, myFrameNumber)
        while myFrameNumber < totalFrames:
            retval, img = cap.read()
            img_height, img_width, img_colors = img.shape
            scale = self.screen / img_height
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            wd02 = QtImage(self.screen, img)
            self.ui.Export_view.setImage(wd02)


app = QApplication(sys.argv)
w = MyWindow(None)
w.windowTitle()
w.show()
app.exec_()
