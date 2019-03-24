import os
import csv
import cv2
import numpy as np
import random
import sklearn

from keras.models import Sequential,Model
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
def load_csv():
    lines = []
    with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for line in reader:
            lines.append(line)
    return lines

def generator(samples, batch_size=32):
    num_samples = len(samples)
    # create adjusted steering measurements for the side camera images
    correction = 0.2 # this is a parameter to tune
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                for i in range(3):
                    current_path = 'data/IMG/' + batch_sample[i].split('/')[-1]
                    center_image = cv2.imread(current_path)
                    center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                    center_angle = float(batch_sample[3])                    
                    images.append(center_image)
                    if i==0: #Center Image
                        measurements.append(center_angle)
                    elif i==1: #Left Image
                        measurements.append(center_angle+correction)
                    else: #Right Image
                       measurements.append(center_angle-correction) 
                
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

def define_model(): 
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu")) 
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.summary()      
    return model

def plot_loss(history_object):
    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('result/loss_curve.png')
    
    
lines = []
lines = load_csv()
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = define_model()
model.compile(loss='mse', optimizer='adam')
batch_size = 32*6
history_object = model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples)//batch_size, validation_data = 
    validation_generator,
    nb_val_samples = len(validation_samples)//batch_size, 
    nb_epoch=5, verbose=1)

plot_loss(history_object)

model.save('model.h5')
