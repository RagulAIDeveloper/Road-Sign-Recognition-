
from PyQt5 import QtCore, QtGui, QtWidgets
from keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout
from keras.models import Sequential,load_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import os

data =[]  #input
labels =[]   #output
classes =43

cur_path  = os.getcwd() # to get current directory

Classes ={0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry',
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }
print("Obtaining image and labels...............")

for i in range(classes):
    #join currentpath and trainpath
    path = os.path.join(cur_path,'C://Users//msrag//OneDrive//Desktop//Deep Learning/road_sign/Train/',str(i))
    #convert to list
    images = os.listdir(path)
    for a in images:
        try:
            #image open
            image = Image.open(path+'\\'+a)
            image = image.resize((30,30))
            #covert to array
            image = np.array(image)
            data.append(image)
            labels.append(i)
            print("{0} Loaded".format(a))
        except:
            print("Error loading image")

#coverting list to array
data =np.array(data)
labels = np.array(labels)

#spliting the data
x_train,x_test,y_train,y_test=train_test_split(data,labels,test_size=.2,random_state =42)

#converting the labels into one hot encoding
y_train = to_categorical(y_train,43)
y_test = to_categorical(y_test,43)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(200, 60, 321, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(150, 420, 151, 41))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(150, 500, 151, 41))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(410, 500, 151, 41))
        self.pushButton_3.setObjectName("pushButton_3")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(400, 390, 171, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.textBrowser_2 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_2.setGeometry(QtCore.QRect(410, 420, 151, 41))
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(170, 140, 391, 221))
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.pushButton.clicked.connect(self.loadImage)
        self.pushButton_2.clicked.connect(self.ClassifyFunction)
        self.pushButton_3.clicked.connect(self.trainingFunction)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "           ROAD SIGN RECOGNITION"))
        self.pushButton.setText(_translate("MainWindow", "Browse Image"))
        self.pushButton_2.setText(_translate("MainWindow", "Classify"))
        self.pushButton_3.setText(_translate("MainWindow", "Training"))
        self.label_2.setText(_translate("MainWindow", "     Recongnized Class"))
    def loadImage(self):
        fileName,_ = QtWidgets.QFileDialog.getOpenFileName(None,"Select Image","","Image Files (*.png *.jpg *.jpeg *.bmp);; All Files (*)")
        print('flieName')
        if fileName:# if the user  gives a files
            
            print(fileName)
            self.file = fileName
            pixmap =QtGui.QPixmap(fileName) # setup pixmap with the provided image
            print('flieName5')
            pixmap = pixmap.scaled(self.label_3.width(),self.label_3.height(),QtCore.Qt.KeepAspectRatio) #scale image
            print('flieName55')
            self.label_3.setPixmap(pixmap) # set the pixmap onto the label
            print('flieName66')
            self.label_3.setAlignment(QtCore.Qt.AlignCenter) # align the image center
    def ClassifyFunction(self):
        #loaded model
        model = load_model("my_model_new.h5")
        print("Loaded model from disk")
        path2 = self.file
        test_image = Image.open(path2)
        test_image = test_image.resize((30,30))
        test_image = np.expand_dims(test_image,axis =0)
        test_image = np.array(test_image)
        #predict  output
        result = model.predict(test_image) [0]
        predicted_class_index =result.argmax()
        sign = Classes[predicted_class_index]
        print(sign)
        self.textBrowser_2 .setText(sign)


    def trainingFunction (self):
        self.textBrowser_2.setText("Training.....")
        model = Sequential()
        model.add(Conv2D(filters =32, kernel_size =(5,5),activation = "relu",input_shape = x_train.shape[1:]))
        model.add(Conv2D(filters =32, kernel_size =(5,5),activation = "relu"))
        model.add(MaxPool2D(pool_size =(2,2)))
        model.add(Dropout(rate =.25))
        model.add(Conv2D(filters =64, kernel_size =(3,3),activation = "relu"))
        model.add(Conv2D(filters =64, kernel_size =(3,3),activation = "relu"))
        model.add(MaxPool2D(pool_size =(2,2)))
        model.add(Dropout(rate =.25))
        model.add(Flatten())
        model.add(Dense(256,activation ="relu"))
        model.add(Dropout(rate =.5))
        model.add(Dense(43,activation ="softmax"))

        print("Initialized model")

        model.compile(loss ="categorical_crossentropy",optimizer = "adam", metrics=["accuracy"])
        #model training
        model.fit(x_train,y_train,batch_size=64,epochs=6,validation_data=(x_test,y_test))
        #model saved
        model.save("my_model_new.h5")
        print("done")
        
        




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
