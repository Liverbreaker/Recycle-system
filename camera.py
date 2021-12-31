# -*- coding: utf-8 -*-
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
import time 
import cv2
import numpy as np
import tensorflow as tf

# get the reference to the webcam
camera = cv2.VideoCapture(0)
camera_height = 500

width = 300
height = 300

#model = load_model('C:/Users/user/anaconda3/files/st6ob_detection.model')
#model = load_model('C:/Users/user/anaconda3/files/object_detection_model.h5')
#model = load_model('C:/Users/user/anaconda3/files/object_detection_model2.h5')
model = load_model('C:/Users/willy/Desktop/recong/model/NoGlass_Epoch_200-400.h5')


class_names = ['bottle','metal','paper']#標籤名稱

base_path = "C:/Users/user/anaconda3/dataset-resized4/dataset-resized2" #資料集路徑

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)#畫面寬
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)#畫面高
#camera.set(cv2.CAP_PROP_FPS, 60)

while(True):
    # read a new frame
    #使用read方法取回影像
    status, frame = camera.read()
    
    #讀取影像失敗
    if not status:
        print("Could not read frame")
        exit()
    
    # flip the frame
    # 橫向翻轉 frame
    # 設0是縱向、設-1是橫向縱向同時翻轉
    frame = cv2.flip(frame, 1)

    # rescaling camera output
    #重新縮放相機輸出
    aspect = frame.shape[1] / float(frame.shape[0])#frame.shape[0]:影像的垂直尺寸
                                                   #frame.shape[1]:影像的水平尺寸
    res = int(aspect * camera_height) # landscape orientation - wide image
    frame = cv2.resize(frame, (res, camera_height))#frame的縮放
                                                   #res(影像的寬),camera_height(影像的高)

    # add rectangle(blue box)
    #cv2.rectangle(影像, 頂點座標,   對象頂點座標,  顏色,       線條寬度)
    cv2.rectangle(frame, (300, 75), (650, 425), (240, 100, 0), 5)

    # get ROI
    roi = frame[75+5:425-5, 300+5:650-5]#裁切區域，要剪掉寬度，所以減5
    
    # parse BRG to RGB
    # cv2.cvtColor:顏色空間轉換
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)#cv2.COLOR_BGR2RGB:將 BGR 轉為 RGB 

    # resize
    roi = cv2.resize(roi, (width, height))#重新縮放
    
    # predict!
    roi_X = tf.expand_dims(roi, axis=0)
    
    preds = model.predict(roi_X)#預測
    ans = class_names[np.argmax(preds[0])]#返回最大值索引號

    # add text
    #cv2.putText(影像, 文字, 文字座標, 文字字形, 文字大小, 文字顏色, 文字粗細)
    ans_text = '{}'.format(ans)
    cv2.putText(frame, 'The classification of the object is : ', (35, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=3)
    cv2.putText(frame, ans_text, (35, 170), 
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255), thickness=5)
    #cv2.drawText(frame, '天主教輔仁大學 資訊管理學系 第三十八屆專題發表 垃圾物件分類智慧辨識裝置', (35, 370), 
               # cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=1)


    # show the frame
    cv2.imshow("Test out", frame)#顯示畫面

    key = cv2.waitKey(1)

    # quit camera if 'q' key is pressed
    #按q關閉影像畫面
    if key & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()