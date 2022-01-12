import os
import csv
import cv2
import numpy as np

lines = []

with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

        
from sklearn.model_selection import train_test_split
import sklearn
lines = lines[1:]
train_lines, validation_lines = train_test_split(lines, test_size=0.2)

def generator(lines, batch_size=32):
    num_lines = len(lines)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(lines)
        for offset in range(0, num_lines, batch_size):
            batch_samples = lines[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                source_path_center = batch_sample[0]
                filename_center = source_path_center.split('/')[-1]
                source_path_left = batch_sample[1]
                filename_left = source_path_left.split('/')[-1]
                source_path_right = batch_sample[2]
                filename_right = source_path_right.split('/')[-1]

                img_center = cv2.imread('data/IMG/' + filename_center)
                img_left = cv2.imread('data/IMG/' + filename_left)
                img_right = cv2.imread('data/IMG/' + filename_right)

                images.append(img_center)
                images.append(img_left)
                images.append(img_right)

                steering_center = float(line[3])
                correction = 0.2
                steering_left = steering_center + 0.2
                steering_right = steering_center - 0.2

                angles.append(steering_center)
                angles.append(steering_left)
                angles.append(steering_right)
            
            augmented_images = []
            augmented_angles = []

            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image,1))
                augmented_angles.append(angle*-1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_lines, batch_size=batch_size)
validation_generator = generator(validation_lines, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
#crop 70 row pixels from the top, 20 row pixels from the bottom, 0 columns from the left and 0 columns from the right
model.add(Cropping2D(cropping=((70,25),(0,0))))
#Convolutional feature map 24@
model.add(Conv2D(filters=24, kernel_size=5, strides=(2,2), activation='relu', padding= 'same'))
#Convolutional feature map 36@
model.add(Conv2D(filters=36, kernel_size=5, strides=(2,2), activation='relu', padding= 'same'))
#Convolutional feature map 48@
model.add(Conv2D(filters=48, kernel_size=5, strides=(2,2), activation='relu', padding= 'same'))
#Convolutional feature map 24@
model.add(Conv2D(filters=64, kernel_size=3, strides=(3,3), activation='relu', padding= 'same'))
#Convolutional feature map 24@
model.add(Conv2D(filters=64, kernel_size=3, strides=(3,3), activation='relu', padding= 'same'))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

import math
from keras.callbacks import ModelCheckpoint, EarlyStopping

save_path = 'model.h5'

checkpoint = ModelCheckpoint(filepath=save_path, monitor='val_loss', save_best_only=True)
stopper = EarlyStopping(monitor='val_loss', min_delta=0.0003, patience=2)

history_object = model.fit_generator(train_generator,
            steps_per_epoch=math.ceil(len(train_lines)/batch_size),
            validation_data=validation_generator,
            validation_steps=math.ceil(len(validation_lines)/batch_size),
            epochs=10, verbose=1, callbacks= [checkpoint, stopper])


import matplotlib.pyplot as plt

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
