
"""
Created on Fri Jul  3 19:58:19 2020

@author: hemes
"""
import cv2
import numpy as np

from keras.models import load_model

classi = load_model('modelhemu.h5') #The CNN model after being trained in cnn.py is saved,
#which is loaded in here for test the model
classi.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
img = cv2.imread(r'C:\Users\hemes\Desktop\cat.jpg')#A cat image which
#is not trained in the model is given as input to predict.

#resize the image according to the convolution layer
img = cv2.resize(img,(64,64))
img = np.reshape(img,[1,64,64,3])
pred = classi.predict_classes(img)
if(pred[0] == 1):
    print("dog")
else:
    print("cat")

