
import cv2
import numpy as np
import  keras
from keras.models import load_model

#加载级联分类器模型
CASE_PATH = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASE_PATH)

#加载卷积神经网络模型
face_recognition_model = keras.Sequential()
MODEL_PATH = 'face_model.h5'
face_recognition_model = load_model(MODEL_PATH)
