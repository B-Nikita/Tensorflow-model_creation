#!/usr/bin/env python
# coding: utf-8

# In[12]:


# set the matplotlib backend so figures can be saved in the background
import matplotlib as plt
plt.use("Agg")

import logging
logging.getLogger('tensorflow').setLevel(logging.CRITICAL)


# In[4]:



from models import minigooglenet_functional
from models import MiniVGGNetModel
from models import shallownet_sequential

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import numpy as np
import argparse


# In[9]:


#construct the argument parser and parse the arguamnts
'''
Two command line arguments include:

- --model : one of the given choices
    choices=`['sequential','functional','class']`
- --plot : The path to the output plot image file.
'''
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="sequential",choices=["sequential", "functional", "class"],help="type of model architecture")
ap.add_argument("-p", "--plot", type=str, required=True,help="Path to plot the output")
args = vars(ap.parse_args())
'''
ap=argparse.ArgumentParser()
ap.add_argument('-m','--model',type=str,default='sequential',
               choices=['sequential','functional','class'],
               help='Type of model architecture')
ap.add_argument('-p','--plot',type=str,required=True,
                help='..\kerasModelCreation\output')
args=vars(ap.parse_args())
'''


# Now remaining tasks are:
# 1. Intialize a number of hyperparameters
# 2. Prepare our data
# 3. Construct data augmentation object

# In[8]:


# initialize the initial learning rate, batch size, and number of epochs to train for
INIT_LR = 1e-2
BATCH_SIZE = 128
NUM_EPOCHS = 60

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog","frog", "horse", "ship", "truck"]

# load the CIFAR-10 dataset
print("[INFO] loading CIFAR-10 dataset...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()

# scale the data to the range [0, 1]
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# convert the labels from integers to vectors
lb=LabelBinarizer()
trainY=lb.fit_transform(trainY)
testY=lb.transform(testY)

#construct the image generator for data augmentation
aug= ImageDataGenerator(rotation_range=18,zoom_range=0.15,
                      width_shift_range=0.2,height_shift_range=0.2,
                      shear_range=0.15,horizontal_flip=True,fill_mode='nearest')


# ### Instantiate the model

# In[11]:


#Check to see if we are using a keras sequntial model
if args['model']=='sequential':
    # instantiate a Keras Sequential model
    print("[INFO] using sequential model...")
    model = shallownet_sequential(32, 32, 3, len(labelNames))
    
elif args['model']=='functional':
    # instantiate a Keras Functional model
    print("[INFO] using functional model...")
    model = minigooglenet_functional(32, 32, 3, len(labelNames))
    
elif args['model']=='class':
    # instantiate a Keras Model sub-class model
    print("[INFO] using model sub-classing...")
    model = MiniVGGNetModel(len(labelNames))
 


# In[10]:


#intialize the optimizer and compile the model
opt=SGD(lr=INIT_LR,momentum=0.9,decay=INIT_LR/NUM_EPOCHS)
print('[INFO] training network..')
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

#train the network using keras' fit_generator method to handle data augmentation
H=model.fit_generator(aug.flow(trainX,trainY,batch_size=BATCH_SIZE),
                     validation_data=(testX,testY),
                     steps_per_epoch=trainX.shape[0]//BATCH_SIZE,
                     epochs=NUM_EPOCHS,
                     verbose=1)


# ### Evaluating model our history and plotting the training history

# In[ ]:


#evaluate the network
print('[INFO] evaluating network...')
predictions=model.predict(testX,batch_size=BATCH_SIZE)
print(classification_report(testY.argmax(axis=1),
                           predictions.argmax(axis=1),
                           target_names=labelNames))

#determine the number of epochs and then construct the plot title
N=np.arange(0,NUM_EPOCHS)
title='Training Loss and Accuracy on CIFAR-10 ({})'.format(args['model'])

#Plot the training loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(N,H.history['loss'],label='train_loss')
plt.plot(N,H.history['val_loss'],label='val_loss')
plt.plot(N,H.history['accuracy'],label='train_acc')
plt.plot(N,H.history['val_accuarcy'],label='val_acc')
plt.title(title)
plt.xlabel('RPOCH #')
plt.ylabel('Loss/Accuarcy')
plt.legend()
plt.savefig(args['plot'])


# In[ ]:




