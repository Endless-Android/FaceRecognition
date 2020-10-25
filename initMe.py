import cv2
import os

CASE_PATH = "haarcascade_frontalface_default.xml"
RAW_IMAGE_DIR = "me/"  #生图路径，即还没处理过的图片
DATASET_DIR = "jm/"   #Yale大学的数据集
face_cascade  = cv2.CascadeClassifier(CASE_PATH)


count = 166
image_list = os.listdir(RAW_IMAGE_DIR)
for image_path in image_list:
    image = cv2.imread(RAW_IMAGE_DIR + image_path)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(5,5))
    for (x,y,width,height) in faces:
        ima = image[y:y + height, x:x + width]
        cv2.imwrite('%ss%d.bmp'%(DATASET_DIR,count), ima)
    count+=1
