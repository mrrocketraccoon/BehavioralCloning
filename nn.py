import csv
import cv2
import numpy as np

lines = []

with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines[1:]:
    source_path_center = line[0]
    filename_center = source_path_center.split('/')[-1]
    source_path_left = line[1]
    filename_left = source_path_left.split('/')[-1]
    source_path_right = line[2]
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
    
    measurements.append(steering_center)
    measurements.append(steering_left)
    measurements.append(steering_right)

#data augmentation, so that the NN learns to turn counter clock-wise as well
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)


X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
#crop 50 row pixels from the top, 20 row pixels from the bottom, 0 columns from the left and 0 columns from the right
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
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5, verbose=1)

model.save('model.h5')

print(history_object.history.keys())

import matplotlib.pyplot as plt

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
