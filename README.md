# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/architecture.png "Model Visualization"
[image2]: ./examples/history.png "Loss history"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* nn_nogenerator.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py dropout_no_generator.h5
```

#### 3. Submission code is usable and readable

The nn_nogenerator.py file contains the code for training and saving the convolutional neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
I have used Nvidia's architecture from their paper End to End Learning for Self-Driving Cars [[1]](#1). The model is comprised of one normalization layer, one cropping layer, five convolutional layers, one flattening layer, five dropout layers and five dense layers. The architecture will be fully covered in the next sections.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (nn_nogenerator.py lines 73-81). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (nn_nogenerator.py line 94). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (nn_nogenerator.py line 84).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of clockwise center lane driving and counter-clockwise center lane driving. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to progressively make the algorithm more robust.

My first step was to use a flattening layer followed by a fully connected layer, after making sure that the pipeline worked I moved to a more complex architecture (LeNet), after making sure the pipeline worked properly and the training was improving, I decided to use the architecture provided from the aforementioned paper by Nvidia. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it included five dropout layers between the five fully connected layers. I also included two callbacks, one model checkpoint to always save the model that delivers the best validation loss, and one for early stopping to finish the training when the validation loss stops improving.


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, to improve the driving behavior in these cases, I collected more data for these situations.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of the following layers:
```sh
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 33, 160, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 17, 80, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 9, 40, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 14, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 5, 64)          36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 320)               0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 320)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 1164)              373644    
_________________________________________________________________
dropout_2 (Dropout)          (None, 1164)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 100)               116500    
_________________________________________________________________
dropout_3 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 50)                5050      
_________________________________________________________________
dropout_4 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 10)                510       
_________________________________________________________________
dropout_5 (Dropout)          (None, 10)                0         
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 11        
=================================================================
Total params: 627,063
Trainable params: 627,063
Non-trainable params: 0
_________________________________________________________________
```

This architecture is based on the paper End to End Learning for Self-Driving Cars [[1]](#1)

![alt text][image1]

CNN Architecture from the paper End to End Learning for Self-Driving Cars.

<br/>

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Then I recorded two laps on the track in a counter-clockwise manner, then I recorded more data for situations where the car would leave the track. I also collected data from maneuvers where the vehicle would return to center lane driving after leaving the center.

To augment the data set, I also flipped images and angles since this would double the dataset. I also used data from the left and right cameras and modified the steering angle by adding a positive and a negative offset respectively, so that the car turn when drifting from the center.

I then preprocessed this data by normalizing the images and cropping them to only include the relevant parts of the road that are needed for learning.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. We can observe from the plot that the lowest MSE loss is obtained after epoch 3. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image2]
## References
<a id="1">[1]</a> 
Bojarski et al. (2016). 
End to End Learning for Self-Driving Cars.