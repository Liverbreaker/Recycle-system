from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import glob
import argparse
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import datetime

#base_path = "C:/Users/user/anaconda3/dataset-resized2/dataset-resized2" #資料集路徑
data_path = "C:/Users/win10/Desktop/recong/data" #資料集路徑
# img_list = glob.glob(os.path.join(base_path, '*/*.jpg'))

#ImageDataGenerator():利用現有的資料經過旋轉、翻轉、縮放等方式增加更多的訓練資料
train_datagen = ImageDataGenerator(                      
    rescale=1./225, shear_range=0.1, zoom_range=0.1,       
    width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, 
    vertical_flip=True, validation_split=0.2)

# test_datagen = ImageDataGenerator(
#     rescale=1./255, validation_split=0.1)

#flow_from_directory(路徑, 圖像被縮放成(300,300), 一次訊量需要的樣本數):以資料夾路徑視為參數,
#                    返回標籤數組的形式, , 可選參數_打亂數據進行和進行變換時的隨機變數種子)
train_generator = train_datagen.flow_from_directory(
    data_path, target_size=(300, 300), batch_size=32,           
    class_mode='categorical', subset='training', seed=0)          

validation_generator = train_datagen.flow_from_directory(
    data_path, target_size=(300, 300), batch_size=32,
    class_mode='categorical', subset='validation', seed=0)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())

print(labels)

def create_model():
    #model_1
    #範例模型
    model = Sequential([
    #第一個卷基層
    Conv2D(filters=32,                 #32個神經元，輸出32張特徵圖
           kernel_size=3,              #卷積核尺寸3x3
           padding='same',             #使用填補法
           activation='relu',          #啟動函數
           input_shape=(300, 300, 3)), #輸入圖片尺寸為300x300(3channals)
    #第一個最大池化層
    MaxPooling2D(pool_size=2),         #檢視視窗2x2
    
    #第二個卷基層
    Conv2D(filters=64,                 #64個神經元，輸出32張特徵圖
           kernel_size=3,              #卷積核尺寸3x3
           padding='same',             #使用填補法
           activation='relu'),         #啟動函數
    
    #第二個最大池化層
    MaxPooling2D(pool_size=2),         #檢視視窗2x2

    #第三個卷基層
    Conv2D(filters=32,                 #32個神經元，輸出32張特徵圖
           kernel_size=3,              #卷積核尺寸3x3
           padding='same',             #使用填補法
           activation='relu'),         #啟動函數
    
    #第三個最大池化層
    MaxPooling2D(pool_size=2),         #檢視視窗2x2
 
    #第四個卷基層
    Conv2D(filters=32,                 #32個神經元，輸出32張特徵圖
           kernel_size=3,              #卷積核尺寸3x3
           padding='same',             #使用填補法
           activation='relu'),         #啟動函數
    
    #第四個最大池化層
    MaxPooling2D(pool_size=2),         #檢視視窗2x2
    #展平層
    Flatten(),                         #將特徵圖拉平
    
    #密集層
    Dense(64, activation='relu'),      #64個神經元
    #密集層
    Dense(3, activation='softmax')     #3個神經元
    ])

    #模型編譯 (損失函數, 優化函數(優化器), 用精準度去評估模型的一個指標)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#創造模型
model = create_model()
#模型總結(摘要)
model.summary()

checkpoint_path = "C:/Users/win10/Desktop/recong/checkpoints/training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# 保存權重回調
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 period=1
                                                 )
model.save_weights(checkpoint_path.format(epoch=0))

# Tensoeboard回調
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="C:/Users/win10/Desktop/recong/logs/NoGlass", histogram_freq=1)

#模型訓練(訓練生成器, 週期, 圖像量除以batch_size(批次大小), 驗證資料生成器, )
EPOCHS = 200
model.fit(
train_generator,
epochs=EPOCHS,
validation_data=validation_generator,
callbacks=[tensorboard_callback,cp_callback])

#模型儲存
model.save('C:/Users/win10/Desktop/recong/model/NoGlass_Epoch_200.h5')
