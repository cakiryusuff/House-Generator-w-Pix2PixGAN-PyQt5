# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 10:56:27 2023

@author: cakir
"""

from pix2pix_model.pix2pix_generator import Generator
from structure import Ui_MainWindow
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPixmap, QImage
import sys
from PyQt5.QtWidgets import QMainWindow
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision

class main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available else "cpu"
        self.qtTasarim = Ui_MainWindow()
        self.mainWindow = QMainWindow()
        self.img = np.zeros((512, 512, 3), np.uint8)
        self.img[:, :, 2] = 170
        
        self.qtTasarim.setupUi(self)

        self.qtTasarim.label.mouseMoveEvent = self.mouse_press_event
        self.qtTasarim.label.mouseReleaseEvent = self.mouse_release_event
        
        self.netG = Generator().to(self.device)
        self.netG.load_state_dict(torch.load("weights/pix2pixGenState_Dictn160.pt"))
        
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.pressed = 0
        self.released = 0
        self.show()
        self.color = (0, 0, 255)
        self.lock = 0
        self.image_for_model = None
        
        self.buttons = [self.qtTasarim.pushButton_14,
                        self.qtTasarim.pushButton_13, self.qtTasarim.pushButton_12,
                        self.qtTasarim.pushButton_11, self.qtTasarim.pushButton_10,
                        self.qtTasarim.pushButton_3, self.qtTasarim.pushButton_8,
                        self.qtTasarim.pushButton_7, self.qtTasarim.pushButton_6,
                        self.qtTasarim.pushButton_5, self.qtTasarim.pushButton_4,]
        
        #self.qtTasarim.pushButton_10.clicked.connect(self.butona_tiklandi)
        for button in self.buttons:
            button.clicked.connect(self.clicked_button)
        
        self.qtTasarim.pushButton_2.clicked.connect(self.clean_label)
        
        self.qtTasarim.pushButton.clicked.connect(self.convert)
    
        
    def clicked_button(self):
        sender = self.sender()
        if sender.text() == 'Facade':
            self.color = (0, 0, 255)
        elif sender.text() == "Molding":
            self.color = (255, 85, 0)
        elif sender.text() == "Cornice":
            self.color = (0, 255, 255)
        elif sender.text() == "Pillar":
            self.color = (255, 0, 0)
        elif sender.text() == "Window":
            self.color = (0, 85, 255)
        elif sender.text() == "Door":
            self.color = (0, 170, 255)
        elif sender.text() == "Sill":
            self.color = (85, 255, 170)
        elif sender.text() == "Blind":
            self.color = (255, 255, 0)
        elif sender.text() == "Balcony":
            self.color = (170, 255, 85)
        elif sender.text() == "Shop":
            self.color = (170, 0, 0)
        elif sender.text() == "Deco":
            self.color = (255, 170, 0)
        
    def paintEvent(self, event):
        if self.start_x is not None and self.start_y is not None and self.end_x is not None and self.end_y is not None:
            width = abs(self.end_x - self.start_x)
            height = abs(self.end_y - self.start_y)
            x = min(self.start_x, self.end_x)
            y = min(self.start_y, self.end_y)
            cv2.rectangle(self.img, (x, y), (x + width, y + height), self.color, thickness = -1)
        
            self.end_x, self.end_y, self.start_x, self.start_y = 0, 0, 0, 0
        
        self.image_for_model = self.img
        height, width, channel = self.img.shape
        q_image = QImage(self.img.data, width, height, channel * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.qtTasarim.label.setPixmap(QPixmap(pixmap))
        
    def mouse_press_event(self, event):
        #if event.button() == Qt.LeftButton:
        if not self.pressed:
            self.start_x = event.x()
            self.start_y = event.y()
            self.end_x = None
            self.end_y = None
            self.pressed = 1
            self.released = 0
            
    def mouse_release_event(self, event):
        if not self.released:
            self.end_x = event.x()
            self.end_y = event.y()
            self.released = 1
            self.pressed = 0
            
    
    def clean_label(self):
        self.img = np.zeros((512, 512, 3), np.uint8)
        self.img[:, :, 2] = 170
        
        height, width, channel = self.img.shape
        q_image = QImage(self.img.data, width, height, channel * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.qtTasarim.label.setPixmap(QPixmap(pixmap))
        
    def convert(self):
        transform_h = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                  ])
        
        self.image_for_model = transform_h(self.image_for_model)
        
        self.image_for_model = self.image_for_model.to(self.device)
        
        self.image_for_model = self.image_for_model.unsqueeze(0)

        predicted = self.netG(self.image_for_model)
        
        predicted = torchvision.utils.make_grid(predicted)
        
        predicted = predicted.cpu().detach().numpy()
        
        predicted = ((predicted + 1) * 127.5).astype(np.uint8)
        
        img = np.transpose(predicted, (1, 2, 0)).copy()
        
        height, width, channel = img.shape

        q_image = QImage(img.data, width, height, width * channel, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.qtTasarim.label_2.setPixmap(pixmap)
    
if __name__ == '__main__':
    App = QApplication(sys.argv)
    window = main()
    window.show()
    sys.exit(App.exec_())