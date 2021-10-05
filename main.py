import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
from PyQt5.uic import loadUi
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtWidgets
from prepare import *
import numpy as np
import face_recognition
import cv2
import os
import sys, datetime
from time import sleep
import glob
import shutil
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.models import load_model

# ImageUpdate = pyqtSignal(QImage)
# ImageUpdate = pyqtSignal()



############################ Home ##########################################
class Home(QtWidgets.QMainWindow):
    def __init__(self):
        super(Home, self).__init__()
        loadUi("./screens/home.ui", self)        
        self.home.clicked.connect(self.goToHome)
        self.stats.clicked.connect(self.goToBasic)        
        self.register_2.clicked.connect(self.goToRegister)
        self.pro.clicked.connect(self.goToPro)
        self.users_list.clicked.connect(self.goToUsers)
        
        # self.worker1 = Worker1()
        # self.worker1.start()
        # self.worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        # ImageUpdate.connect(self.ImageUpdateSlot)
    
    def ImageUpdateSlot(self, Image):
        self.frameVideo.setPixmap(QPixmap.fromImage(Image))

    def goToHome(self):
        home = Home()
        widget.addWidget(home)
        widget.setCurrentIndex(widget.currentIndex()+1)

    def goToBasic(self):
        stats = Basic()
        widget.addWidget(stats)
        widget.setCurrentIndex(widget.currentIndex()+1)
        
    def goToPro(self):
        pro = Pro()
        widget.addWidget(pro)
        widget.setCurrentIndex(widget.currentIndex()+1)
    
    def goToRegister(self):
        reg = Register()
        widget.addWidget(reg)
        widget.setCurrentIndex(widget.currentIndex()+1)
            
    def goToUsers(self):
        u =Users()
        widget.addWidget(u)
        widget.setCurrentIndex(widget.currentIndex()+1)

################################# Pro #######################################
class Pro(QtWidgets.QMainWindow):
    def __init__(self):
        super(Pro, self).__init__()
        loadUi("./screens/pro.ui", self)
        self.home.clicked.connect(self.goToHome)
        self.stats.clicked.connect(self.goToBasic)
        self.register_2.clicked.connect(self.goToRegister)
        self.pro.clicked.connect(self.goToPro)
        self.users_list.clicked.connect(self.goToUsers)
        self.tableWidget.setColumnWidth(0,200)
        self.tableWidget.setColumnWidth(1,200)
        self.loadTableData()
    
    def loadTableData(self):
        with open('database.csv', "r") as f:
            data = f.readlines()
            self.tableWidget.setRowCount(len(data))
            row = 0
            for i in range(len(data)):
                temp = data[i].split(',')
                self.tableWidget.setItem(row, 1, QtWidgets.QTableWidgetItem(temp[0]))
                self.tableWidget.setItem(row, 0, QtWidgets.QTableWidgetItem(temp[1]))
                row += 1
    
    def goToHome(self):
        home = Home()
        widget.addWidget(home)
        widget.setCurrentIndex(widget.currentIndex()+1)

    def goToBasic(self):
        stats = Basic()
        widget.addWidget(stats)
        widget.setCurrentIndex(widget.currentIndex()+1)    
        
    def goToPro(self):
        pro = Pro()
        widget.addWidget(pro)
        widget.setCurrentIndex(widget.currentIndex()+1)

    def goToRegister(self):
        reg = Register()
        widget.addWidget(reg)
        widget.setCurrentIndex(widget.currentIndex()+1)
            
    def goToUsers(self):
        u =Users()
        widget.addWidget(u)
        widget.setCurrentIndex(widget.currentIndex()+1)

################################# basic #######################################
class Basic(QtWidgets.QMainWindow):
    def __init__(self):
        super(Basic, self).__init__()
        loadUi("./screens/basic.ui", self)
        self.home.clicked.connect(self.goToHome)
        self.stats.clicked.connect(self.goToBasic)
        self.register_2.clicked.connect(self.goToRegister)
        self.pro.clicked.connect(self.goToPro)
        self.users_list.clicked.connect(self.goToUsers)
        self.tableWidget.setColumnWidth(0,200)
        self.tableWidget.setColumnWidth(1,200)
        self.loadTableData()
    
    def loadTableData(self):
        with open('database1.csv', "r") as f:
            data = f.readlines()
            self.tableWidget.setRowCount(len(data))
            row = 0
            for i in range(len(data)):
                temp = data[i].split(',')
                self.tableWidget.setItem(row, 0, QtWidgets.QTableWidgetItem(temp[0]))
                self.tableWidget.setItem(row, 1, QtWidgets.QTableWidgetItem(temp[1].replace("\n", "")))
                row += 1
    
    def goToHome(self):
        home = Home()
        widget.addWidget(home)
        widget.setCurrentIndex(widget.currentIndex()+1)

    def goToBasic(self):
        stats = Basic()
        widget.addWidget(stats)
        widget.setCurrentIndex(widget.currentIndex()+1)
        
    def goToPro(self):
        pro = Pro()
        widget.addWidget(pro)
        widget.setCurrentIndex(widget.currentIndex()+1)

    def goToRegister(self):
        reg = Register()
        widget.addWidget(reg)
        widget.setCurrentIndex(widget.currentIndex()+1)
            
    def goToUsers(self):
        u =Users()
        widget.addWidget(u)
        widget.setCurrentIndex(widget.currentIndex()+1)
        

################################3 usersList ##################################
class Users(QtWidgets.QMainWindow):
    def __init__(self):
        super(Users, self).__init__()
        loadUi("./screens/userlist.ui", self)
        self.home.clicked.connect(self.goToHome)
        self.stats.clicked.connect(self.goToBasic)
        self.register_2.clicked.connect(self.goToRegister)
        self.pro.clicked.connect(self.goToPro)
        self.users_list.clicked.connect(self.goToUsers)
        self.tableWidget.setColumnWidth(0,200)
        self.tableWidget.setColumnWidth(1,200)
        self.loadTableData()
    
    def loadTableData(self):
        with open('users.csv', "r") as f:
            data = f.readlines()
            self.tableWidget.setRowCount(len(data))
            row = 0
            for i in range(len(data)):
                temp = data[i].split(',')
                self.tableWidget.setItem(row, 0, QtWidgets.QTableWidgetItem(temp[0]))
                self.tableWidget.setItem(row, 1, QtWidgets.QTableWidgetItem(temp[1]))
                self.tableWidget.setItem(row, 2, QtWidgets.QTableWidgetItem(temp[2].replace("\n", "")))
                row += 1
    
    def goToHome(self):
        home = Home()
        widget.addWidget(home)
        widget.setCurrentIndex(widget.currentIndex()+1)

    def goToBasic(self):
        stats = Basic()
        widget.addWidget(stats)
        widget.setCurrentIndex(widget.currentIndex()+1)
        
    def goToPro(self):
        pro = Pro()
        widget.addWidget(pro)
        widget.setCurrentIndex(widget.currentIndex()+1)

    def goToRegister(self):
        reg = Register()
        widget.addWidget(reg)
        widget.setCurrentIndex(widget.currentIndex()+1)
        
    def goToUsers(self):
        u =Users()
        widget.addWidget(u)
        widget.setCurrentIndex(widget.currentIndex()+1)

################################3 Register ##################################
class Register(QtWidgets.QMainWindow):
    def __init__(self):
        super(Register, self).__init__()
        loadUi("./screens/register.ui", self)
        self.home.clicked.connect(self.goToHome)
        self.stats.clicked.connect(self.goToBasic)
        self.browse.clicked.connect(self.browseFile)
        self.register_user.clicked.connect(self.registerUser)
        self.pro.clicked.connect(self.goToPro)
        self.users_list.clicked.connect(self.goToUsers)
        self.filePath = ""

    def registerUser(self):
        name = self.name.text()
        email = self.email.text()
        post = self.post.text()
        if self.filePath == "":
            print("Please select photo")
        else:
            print(f"Name: {name}, Email: {email}, Designation: {post}")
            f = open("users.csv", "a")
            f.write(f"{name},{email},{post}\n")
            f.close()

            src_dir = self.filePath
            dst_dir = "./registered_users"
            for jpgfile in glob.iglob(src_dir):
                shutil.copy(jpgfile, dst_dir)
        
    
    def browseFile(self):
        fname = QFileDialog.getOpenFileName(self, 'Open File', '')
        self.filePath = fname[0]
        print(fname[0])
        self.photo.setText(fname[0])
    
    def goToHome(self):
        home = Home()
        widget.addWidget(home)
        widget.setCurrentIndex(widget.currentIndex()+1)

    def goToBasic(self):
        stats = Basic()
        widget.addWidget(stats)
        widget.setCurrentIndex(widget.currentIndex()+1)
        
    def goToPro(self):
        pro = Pro()
        widget.addWidget(pro)
        widget.setCurrentIndex(widget.currentIndex()+1)

    def goToRegister(self):
        reg = Register()
        widget.addWidget(reg)
        widget.setCurrentIndex(widget.currentIndex()+1)
        
    def goToUsers(self):
        u =Users()
        widget.addWidget(u)
        widget.setCurrentIndex(widget.currentIndex()+1)

################################ Thread #####################################

class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)

    def run(self):
        self.ThreadActive = True
        capture = cv2.VideoCapture(0)
        while self.ThreadActive:
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument("--interval", "-i", type=int,
                action='store',
                default=5,
                help='Detection interval in seconds, default=3')

            args = parser.parse_args()
            # run(args.interval)
            print("INTERVALS", args.interval)


            video_capture = cv2.VideoCapture(0)
            # address = 'http://192.168.137.179:8080/video'
            # video_capture.open(address)

            # exit if video not opened
            if not video_capture.isOpened():
                print('Cannot open video')
                sys.exit()
            
            # read first frame
            ok, frame = video_capture.read()
            if not ok:
                print('Error reading video')
                sys.exit()

            # init detection pipeline
            pipeline = Pipeline(event_interval=args.interval)

            # hot start detection
            # read some frames to get first detection
            faces = ()
            detected = False
            while not detected:
                _, frame = video_capture.read()
                faces, detected, preds = pipeline.detect_and_track(frame)
                print("hot start; ", faces, type(faces), "size: ", np.array(faces).size)

            # Draw the bounding box around the face
            draw_boxes(frame, faces, preds, True)
            # Check if the person have mask or not. Do this only when detecting the face
            state = "DETECTOR" if detected else "TRACKING"
            if state=="DETECTOR" and (faces and faces != [(0, 0, 0, 0)]) and preds: 
                (mask, withoutMask) = preds[0]
                label = "Mask" if mask > withoutMask-0.10 else "No Mask"
                markTheData1(label)

            if faces != [] and PRO:
                # Get the coordinates of the face
                startX, startY, endX, endY = faces[0]
                endX = endX+startX
                endY = endY + startY
                faces = [(startY, endX, endY, startX)]
                (mask, withoutMask) = preds[0]
                # If the person has not mask recognize its face
                if not (mask > withoutMask-0.10):
                    label = face_recognize(frame, faces)
                    cv2.putText(frame, label, (startX, endY + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 2)
                if state=="DETECTOR" and (faces and faces != [(0, 0, 0, 0)]) and preds: 
                    (mask, withoutMask) = preds[0]
                    # Add an entry to the database
                    if not (mask > withoutMask-0.10):
                        markTheData(label, mask > withoutMask-0.10)

            
            ##
            ## main loop
            ##
            while True:
                # Capture frame-by-frame
                _, frame = video_capture.read()

                # update pipeline
                boxes, detected_new, preds = pipeline.boxes_for_frame(frame)

                # logging
                state = "DETECTOR" if detected_new else "TRACKING"
                print("[%s] boxes: %s" % (state, boxes))
                if state=="DETECTOR" and (boxes and boxes != [(0, 0, 0, 0)]) and preds: 
                    print(preds)
                    (mask, withoutMask) = preds[0]
                    label = "Mask" if mask > withoutMask-0.10 else "No Mask"
                    markTheData1(label)

                # update screen
                color = color = GREEN if detected_new else BLUE
                draw_boxes(frame, boxes, preds, color)
                if boxes != [] and PRO:
                    draw_boxes(frame, boxes, preds, color)
                    startX, startY, endX, endY = boxes[0]
                    endX = endX+startX
                    endY = endY + startY
                    boxes = [(startY, endX, endY, startX)]

                    (mask, withoutMask) = preds[0]
                    if not (mask > withoutMask-0.10):
                        label = face_recognize(frame, boxes)
                        cv2.putText(frame, label, (startX, endY + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 2)
                    if state=="DETECTOR" and (boxes and boxes != [(0, 0, 0, 0)]) and preds: 
                        (mask, withoutMask) = preds[0]
                        if not (mask > withoutMask-0.10):
                            markTheData(label, mask > withoutMask-0.10)


                # Display the resulting frame
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # FlippedImage = cv2.flip(Image, 1)
                FlippedImage = Image
                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888 )
                pic = ConvertToQtFormat.scaled(500,330)
                # print("TYPE", type(ImageUpdate))
                self.ImageUpdate.emit(pic)
                # cv2.imshow('Video', frame)

                # quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("IN CROSS")
                    break


            # When everything is done, release the capture
            video_capture.release()
            cv2.destroyAllWindows()

            # ret, frame = capture.read()
            # if ret:
            #     Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #     FlippedImage = cv2.flip(Image, 1)
            #     ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888 )
            #     pic = ConvertToQtFormat.scaled(500,330)
            #     self.ImageUpdate.emit(pic)

    def stop(self):
        self.ThreadActive = False
        self.quit()


worker1 = Worker1()
worker1.start()

app = QApplication(sys.argv)
loginWindow = Home()
widget = QtWidgets.QStackedWidget()
widget.addWidget(loginWindow)
widget.setFixedHeight(400)
widget.setFixedWidth(700)
widget.show()

app.exec_()