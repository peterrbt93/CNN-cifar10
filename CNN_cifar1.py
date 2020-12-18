# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 17:23:43 2020

@author: pete_
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

#%%

# =============================================================================
# Recurrent networks are good for temporal correlations, Convolutional
#  NN are good for spatial correlations. Instead of having layers that
#  are fully connected, they are only connected to nodes close to them
#  Effectively scanning the image one segment at a time. Why its known
#  as a filter. (normal NN with only local connect is a 1d convolution)
#  Its esentially a multi dimensional NN. If a picture has mutliple channels
#  i.e. RGB then this is still considered 2D convolution, since the channel
#  space order doesn't matter. (We only slide in HxW dimensions).
#  Image is 
#  If we had time as well, then this is 3D convoluiton.
#  The activation is then maximized at edges for example. Combine this with
#  Pooling layers that reduce the resolution to optimize calculation. 
#  MaxPooling just takes the max value of a segment.
# The kernel/window that scans is fixed for each channel to channel
# i.e. the same weight scans the entire image of previous layer pr channel
#
# Kernel filtering is used in image processing. If weights are equal
# this corresponds to a mean blur. If gaussian distribution in 2D with
# peak at center, this is a gaussian blur. The cool thing is that the
# ML algorithm learns what the important filters are by itself!
# Even filters for edge detection!
#
# CNN corr in space go G(x,x') from physics. RNN corr in time G(t,t')!!!
# [-1,0,1]
# [-2,0,2]  -> Sobel operator (window/kernel like this detects edges in x)
# [-1,0,1]
# Use gaussian blur first before sobel to remove noise
# =============================================================================


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']


#%%

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

#%%

# Conv2D(64, (3, 3)... # Conv with 64 filters (depth) with window/kernel
# size of 3x3) 
# MaxPooling2D((2, 2))) #Max on squares of 2x2

#Weights are (window width, window height, input channels, output channels)
# input channels are for example 3 for RGB input, output channels are
# number of filters (how many features per segment to analyze)

#Conv layers keep reducing image size and increasing features! The features
# are then fed into a standard flat neural net to do classification!


# =============================================================================
# model.output_shape            model.weights.shape
# (None, 32, 32, 3) (input)      
#                               w(3, 3, 3, 32) b(32,)    p=3*3*3*32+32=896
# (None, 30, 30, 32)              
#                               maxpool
# (None, 15, 15, 32)              
#                               w(3, 3, 32, 64) b(64,)   p=18496
# (None, 13, 13, 64)              
#                               maxpool
# (None, 6, 6, 64)                
#                               w(3, 3, 64, 128) b(128,) p=73856
# (None, 4, 4, 128)               
#                               flatten (4*4*128=2048)
# (None, 2048)                    
#                               w(2048, 128) b(128,)     p=262272
# (None, 128)                     
#                               w(128, 10) b(10,)        p=1290
# (None, 10)
# =============================================================================


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10))
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)

#Got from 70%-> 89% on training just by changing hidden layer shapes!
#2x2 = 0.718
#3x3 = 0.7188
#4x4 = 0.7

#%%

plt.figure(figsize=(25,15))
for i in range(35):
    plt.subplot(5,7,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel("True: "+class_names[ test_labels[i][0]] + ", Pred: " +class_names[np.argmax(model.predict(test_images[i].reshape(1,32,32,3)))],color=('red' if test_labels[i][0]!=np.argmax(model.predict(test_images[i].reshape(1,32,32,3))) else 'green'))
plt.show()