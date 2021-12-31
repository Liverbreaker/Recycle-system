from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, Convolution2D
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
import cv2
import os
import glob


base_path = "C:/Users/win10/Desktop/recong/data" #資料集路徑
img_list = glob.glob(os.path.join(base_path, '*/*.jpg'))

train_datagen = ImageDataGenerator(                      
    rescale=1./225, shear_range=0.1, zoom_range=0.1,       
    width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, 
    vertical_flip=True, validation_split=0.2)

# test_datagen = ImageDataGenerator(
#     rescale=1./255, validation_split=0.1)
    
train_generator = train_datagen.flow_from_directory(
    base_path, target_size=(300, 300), batch_size=32,           
    class_mode='categorical', subset='training', seed=0)          

validation_generator = train_datagen.flow_from_directory(
    base_path, target_size=(300, 300), batch_size=32,
    class_mode='categorical', subset='validation', seed=0)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())

print(labels)

model = load_model('C:/Users/win10/Desktop/recong/model/NoGlass_Epoch_200.h5')#11042//256,1225//256

# Summary of model
model.summary()
model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=0.001),
                      metrics=['accuracy'])

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
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="C:/Users/win10/Desktop/recong/logs/NoGlass_200-400", histogram_freq=1)

#model.fit_generator(train_generator, epochs=50, steps_per_epoch=50,validation_data=validation_generator, validation_steps=25)
EPOCHS = 200
model.fit(train_generator,
            epochs=EPOCHS,
            validation_data=validation_generator,
            callbacks=[tensorboard_callback, cp_callback])
                    
model.save('C:/Users/win10/Desktop/recong/model/NoGlass_Epoch_200-400.h5')
