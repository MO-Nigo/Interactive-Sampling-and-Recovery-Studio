from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUiType
import os
from os import path
from PyQt5.QtWidgets import QLabel

import sys
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout

from app_logic import AppLogic


# ##Importing classes
# from classes.loadSignal import loadCSVFile


#Linking the UI to this python file
FORM_CLASS, _ = loadUiType(path.join(path.dirname(__file__), "main.ui"))
# lable_inmax
class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.f_max=0
        self.flag=0
        ## UI Customization
        # Object's Initial Labels Tab1
        self.btnLoad.setText("Load Signal")
        self.btnRemove.setText("Remove Signal")
        self.labelNoise.setText("Noise Level")
        self.labelSignal.setText("Original Signal")
        self.labelSlider.setText("Hz")
        self.labelSample.setText("Sampled Signal")
        self.labelError.setText("Error")
        self.labelRMax.setText("   ")
        self.lable_inmax.setText(" max_frequancy ")
        self.labelMax.setText("Max Frequency(Hz):")
        self.labelSamp.setText("Sampling Freq.")
        self.labelNoise2.setText("%")

        
        # Set text alignment for labels Tab1
        self.labelNoise.setAlignment(Qt.AlignCenter)
        self.labelSignal.setAlignment(Qt.AlignCenter)
        self.labelSlider.setAlignment(Qt.AlignCenter)
        self.labelSample.setAlignment(Qt.AlignCenter)
        self.labelError.setAlignment(Qt.AlignCenter)
        
        # Object's Initial Labels Tab2
        self.btnRemove2.setText("Remove")
        self.btnAdd.setText("Add")
        self.btnCreate.setText("Create")
        self.btnConfirm.setText("Confirm")
        self.labelName.setText("Signal Name")
        self.labelFreq.setText("Frequency")
        self.labelRFreq.setText("Frequency")
        self.labelAmp.setText("Amplitude")
        self.labelPhase.setText("Phase Shift")
        self.labelRRFreq.setText("   ")


        # Set text alignment for labels Tab2
        self.labelName.setAlignment(Qt.AlignCenter)
        self.labelFreq.setAlignment(Qt.AlignCenter)
        self.labelRFreq.setAlignment(Qt.AlignCenter)
        self.labelAmp.setAlignment(Qt.AlignCenter)
        self.labelPhase.setAlignment(Qt.AlignCenter)

        # Setting the app's name and icon
        self.setWindowTitle("Signal Sampler")
        self.setWindowIcon(QIcon('icon.png'))
        #App UI Customization
        self.setStyleSheet('''
            QLabel {
                font-size: 14px;
                color: black;
            }
            QPushButton {
                background-color: white;
                color: black;
                border: 1px solid #CCCCFF;
                border-radius: 5px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #CCCCFF;
                color: black;
            }
            QSlider::handle:horizontal {
                background: #CCCCFF;
                border: 1px solid #CCCCFF;
                width: 20px;
            }
            QSlider::handle:verticle {
                background: #CCCCFF;
                border: 1px solid #CCCCFF;
                width: 20px;
            }
            QComboBox {
                background-color: white;
                border: 1px solid #CCCCFF;
                padding: 1px 18px 1px 3px;
            }
            QComboBox:hover {
                background-color: #CCCCFF;
                border: 1px solid #CCCCFF;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #CCCCFF;
            }
            QComboBox QAbstractItemView {
                background: #CCCCFF;
                border: 1px solid #CCCCFF;
            }
        ''')

        # Launching the app in fullscreen
        self.showMaximized()

        ## End of UI Customization

        ##Triggers
        self.app_logic = AppLogic(self)
        self.btnLoad.clicked.connect(lambda:self.app_logic.load_signal())
        self.btnRemove.clicked.connect(lambda:self.app_logic.remove_signal())  
        self.sliderHz.valueChanged.connect(self.handle_frequancy_slider)
        self.sliderNoise.valueChanged.connect(self.handle_noise_value_change)

        self.btnCreate.clicked.connect(lambda: self.app_logic.create_and_plot_signal())
        self.btnAdd.clicked.connect(lambda: self.app_logic.composer())
        self.btnRemove2.clicked.connect(lambda: self.app_logic.remove_signal_tab2())
        self.btnConfirm.clicked.connect(lambda: self.app_logic.plot_mix())
        ##Plots creation
        
        # Create PyQtGraph widgets for the three frames
        self.plotSignal = pg.PlotWidget()
        self.plotSample = pg.PlotWidget()
        self.plotError = pg.PlotWidget()
        self.plotBefore = pg.PlotWidget()
        self.plotAfter = pg.PlotWidget()

        # Set layouts for the frames
        self.frameSignal.setLayout(QVBoxLayout())
        self.frameSample.setLayout(QVBoxLayout())
        self.frameError.setLayout(QVBoxLayout())
        self.frameBefore.setLayout(QVBoxLayout())
        self.frameAfter.setLayout(QVBoxLayout())

        # Customize the plot appearance
        self.plotSignal.setBackground('w')  # Set white background
        self.plotSignal.getAxis('bottom').setPen('k')  # Set the bottom axis line color to black
        self.plotSignal.getAxis('left').setPen('k')    # Set the left axis line color to black
        self.plotSignal.getAxis('bottom').setTextPen('k')  # Set tick label text color to black
        self.plotSignal.getAxis('left').setTextPen('k')

        self.plotSample.setBackground('w')
        self.plotSample.getAxis('bottom').setPen('k')
        self.plotSample.getAxis('left').setPen('k')
        self.plotSample.getAxis('bottom').setTextPen('k')
        self.plotSample.getAxis('left').setTextPen('k')

        self.plotError.setBackground('w')
        self.plotError.getAxis('bottom').setPen('k')
        self.plotError.getAxis('left').setPen('k')
        self.plotError.getAxis('bottom').setTextPen('k')
        self.plotError.getAxis('left').setTextPen('k')

        self.plotBefore.setBackground('w')
        self.plotBefore.getAxis('bottom').setPen('k')
        self.plotBefore.getAxis('left').setPen('k')
        self.plotBefore.getAxis('bottom').setTextPen('k')
        self.plotBefore.getAxis('left').setTextPen('k')

        self.plotAfter.setBackground('w')
        self.plotAfter.getAxis('bottom').setPen('k')
        self.plotAfter.getAxis('left').setPen('k')
        self.plotAfter.getAxis('bottom').setTextPen('k')
        self.plotAfter.getAxis('left').setTextPen('k')

        # Add the PyQtGraph widgets to the frames
        self.frameSignal.layout().addWidget(self.plotSignal)
        self.frameSample.layout().addWidget(self.plotSample)
        self.frameError.layout().addWidget(self.plotError)
        self.frameBefore.layout().addWidget(self.plotBefore)
        self.frameAfter.layout().addWidget(self.plotAfter)




    #Linking

    def update_labelRMax(self, new_value):
        self.labelRMax.setText(f"{new_value:.2f} Hz")
        self.f_max=new_value
    def update_labelSlider(self, new_value):
        self.labelSlider.setText(f"{str(new_value)} Hz")
        self.lable_inmax.setText(f"{new_value/self.f_max : .2f}   max_frequancy")
    def composer_freq(self,max_value):
        self.labelRRFreq.setText(f"{max_value} Hz")

    def handle_noise_value_change(self):
        if self.sliderNoise.value() != 1:
            self.app_logic.plot_noise_signal()
            self.flag=1
        else:
            self.app_logic.sample_and_plot()
            self.flag=0
    def handle_frequancy_slider(self):
        if self.flag == 0 :
            self.app_logic.sample_and_plot()
        else:
            self.app_logic.plot_noise_signal()




def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()
