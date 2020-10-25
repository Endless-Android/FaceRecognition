import cv2
import os


import numpy as np
#级联分类器
CASE_PATH = "haarcascade_frontalface_default.xml"
RAW_IMAGE_DIR = "me/"  #生图路径，即还没处理过的图片
DATASET_DIR = "jm/"   #Yale大学的数据集
face_cascade  = cv2.CascadeClassifier(CASE_PATH)

def save_faces(img,name,x,y,width,height):
    image = img[y:y+height,x:x+width]
    cv2.imwrite(name,image)

    image_list = os.listdir(RAW_IMAGE_DIR)
    count = 166
    for image_path in image_list:
        image = cv2.imread(RAW_IMAGE_DIR + image_path)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(5,5))
        for (x,y,width,height) in faces:
            save_faces(image,'%ss%d.bmp'%(DATASET_DIR,count),x,y-30,width,height+30)
        count+=1


def resize_without_deformation(image, size = (100, 100)):
    height, width, _ = image.shape
    longest_edge = max(height, width)
    top, bottom, left, right = 0, 0, 0, 0
    if height < longest_edge:
        height_diff = longest_edge - height
        top = int(height_diff / 2)
        bottom = height_diff - top
    elif width < longest_edge:
        width_diff = longest_edge - width
        left = int(width_diff / 2)
        right = width_diff - left

    image_with_border = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = [0, 0, 0])

    resized_image = cv2.resize(image_with_border, size)

    return resized_image

def read_image(size=None):
    data_x,data_y = [],[]
    for i in range(1,177):
        try:
            im = cv2.imread('jm/s%s.bmp'%str(i))
            if size is None:
                size = (100,100)
            im = resize_without_deformation(im,size)
            data_x.append(np.asanyarray(im,dtype=np.int8))
            data_y.append(str(int((i-1)/11.0)))
        except IOError as e:
            print(e)
        except:
            print("Unknown Error!")
    return data_x,data_y



from keras.layers import Conv2D, MaxPooling2D
from keras.layers import  Dense, Dropout, Flatten
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.model_selection import  train_test_split
import keras

IMAGE_SIZE = 100
raw_images, raw_labels = read_image(size=(IMAGE_SIZE, IMAGE_SIZE))
#把图像转换为float类型，方便归一化
raw_images, raw_labels = np.asarray(raw_images, dtype = np.float32), np.asarray(raw_labels, dtype = np.int32)
ont_hot_labels = np_utils.to_categorical(raw_labels)
train_input, valid_input, train_output, valid_output =train_test_split(raw_images,
                  ont_hot_labels,
                  test_size = 0.3)

train_input /= 255.0
valid_input /= 255.0

face_recognition_model = keras.Sequential()
face_recognition_model.add(Conv2D(32, 3, 3, border_mode='valid',
                                  subsample = (1, 1),
                                  dim_ordering = 'tf',
                                  input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3),
                                  activation='relu'))

face_recognition_model.add(Conv2D(32, 3, 3,border_mode='valid',
                                  subsample = (1, 1),
                                  dim_ordering = 'tf',
                                  activation = 'relu'))

face_recognition_model.add(MaxPooling2D(pool_size=(2, 2)))

face_recognition_model.add(Dropout(0.2))

face_recognition_model.add(Conv2D(64, 3, 3, border_mode='valid',
                                  subsample = (1, 1),
                                  dim_ordering = 'tf',
                                  activation = 'relu'))

face_recognition_model.add(Conv2D(64, 3, 3, border_mode='valid',
                                  subsample = (1, 1),
                                  dim_ordering = 'tf',
                                  activation = 'relu'))

face_recognition_model.add(MaxPooling2D(pool_size=(2, 2)))
face_recognition_model.add(Dropout(0.2))

face_recognition_model.add(Flatten())

face_recognition_model.add(Dense(512, activation = 'relu'))

face_recognition_model.add(Dropout(0.4))

face_recognition_model.add(Dense(len(ont_hot_labels[0]), activation = 'sigmoid'))

face_recognition_model.summary()

learning_rate = 0.01
decay = 1e-6
momentum = 0.8
nesterov = True
sgd_optimizer = SGD(lr = learning_rate, decay = decay,
                    momentum = momentum, nesterov = nesterov)

face_recognition_model.compile(loss = 'categorical_crossentropy',
                               optimizer = sgd_optimizer,
                               metrics = ['accuracy'])

batch_size = 20 #每批训练数据量的大小
epochs = 100
face_recognition_model.fit(train_input, train_output,
                           epochs = epochs,
                           batch_size = batch_size,
                           shuffle = True,
                           validation_data = (valid_input, valid_output))

print(face_recognition_model.evaluate(valid_input, valid_output, verbose=0))
MODEL_PATH = 'face_model.h5'
face_recognition_model.save(MODEL_PATH)