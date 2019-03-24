# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./result/loss_curve.png "Loss Curve"
[image2]: ./result/original.png "Normal Image"
[image3]: ./result/flip.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

python drive.py model.h5


#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 24 and 64 (model.py lines 60-76) 

The model includes RELU layers to introduce nonlinearity (code line 64-68), and the data is normalized in the model using a Keras lambda layer (code line 62). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 69). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 103-107). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

This model is accurency. The loss only 1.68% and validation loss only 2.12%.

Epoch 1/5
33/33 [==============================] - 13s - loss: 0.0244 - val_loss: 0.0211
Epoch 2/5
33/33 [==============================] - 8s - loss: 0.0192 - val_loss: 0.0187
Epoch 3/5
33/33 [==============================] - 8s - loss: 0.0210 - val_loss: 0.0227
Epoch 4/5
33/33 [==============================] - 8s - loss: 0.0176 - val_loss: 0.0158
Epoch 5/5
33/33 [==============================] - 8s - loss: 0.0168 - val_loss: 0.0212

[image1]: ./result/loss_curve.png "Loss Curve"

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 101).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

In order to increase the size of training data, I involves flipping images and taking the opposite sign of the steering measurement. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to normailize image, crop it for decreasing data size, then use several 2D convolution to get deep featrues. Flatten these features then use four dense to train the model.

My first step was to use a convolution neural network model similar to the model used in "Even More Powerful Network" this part. I thought this model might be appropriate because it use a lot of convolution to 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that dropout algorithm is used after five convolutions.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. In order to improve the driving behavior in these cases, I use multiple camera data and add flipping images for helping with the left turn bias.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 60-76) consisted of a convolution neural network with the following layers and layer sizes.

_________________________________________________________________

Layer (type)                 Output Shape              Param   
_________________________________________________________________
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928     
_________________________________________________________________
dropout_1 (Dropout)          (None, 1, 33, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               211300    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
_________________________________________________________________
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0
_________________________________________________________________



#### 3. Creation of the Training Set & Training Process

I choose to use a workspace, sample driving data is already included in my files.

To augment the data sat, I also flipped images and angles thinking that this would help for left turn bias For example, here is an image that has then been flipped:

[image2]: ./result/original.png "Normal Image"
[image3]: ./result/flip.png "Flipped Image"


After the collection process, I had (160, 320, 3) number of data points. I then preprocessed this data by cropping. Then it size became (70, 320, 3).


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4. I used an adam optimizer so that manually training the learning rate wasn't necessary.
