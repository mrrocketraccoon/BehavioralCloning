import csv
import cv2
import numpy as np

lines = []

with open('new_data/driving_log.csv') as csvfile:
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
    
    img_center = cv2.imread('new_data/IMG/' + filename_center)
    img_left = cv2.imread('new_data/IMG/' + filename_left)
    img_right = cv2.imread('new_data/IMG/' + filename_right)
#    try:
#        print(img_center.shape)
#    except:
#        print(filename_center)
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

from keras.models import load_model
model = load_model("new_model.h5")
#model.summary()

import math
from keras.callbacks import ModelCheckpoint, EarlyStopping

save_path = 'transfer_model.h5'

my_callbacks = [
    ModelCheckpoint(filepath=save_path, monitor='val_loss', save_best_only=True),
    EarlyStopping(monitor='val_loss', min_delta=0.0003, patience=2)
]
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10, verbose=1, callbacks=my_callbacks)



import matplotlib.pyplot as plt

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
