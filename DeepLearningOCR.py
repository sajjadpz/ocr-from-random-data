#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
import pandas as pd
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from os import listdir
from numpy import array


# In[2]:


def load_photos(directory):
	images = dict()
	for name in listdir(directory):
		if name.endswith(".jpg"):
            # load an image from file
		    filename = directory + '/' + name
		    image = load_img(filename, target_size=(50, 50))
		# convert the image pixels to a numpy array
		    image = img_to_array(image)
		# get image id
		    image_id = name.split('.')[0]
		    images[image_id] = image
	return images
 
# load images
directory = 'out'
#images = load_photos(directory)
#print('Loaded Images: %d' % len(images))


# In[2]:


#read data from csv
data = pd.read_csv("tessdata.csv", names=["image", "text","confidence"])

#images i.e X
X = data['image']
#target variable
y = data['confidence']

data.head(5)


# In[4]:


# def create_data():
#     X_data, Y_data =  list(), list()  
#     for i, j in data.iterrows():
#         image_id, confidence = X[i].split('.')[0], y[i]
#         image = images[image_id]
#         X_data.append(image)
#         Y_data.append(confidence)
#     X_data, Y_data = array(X_data), array(Y_data)
#     return X_data, Y_data


# In[3]:


def create_data():
    X_data, Y_data =  list(), list()  
    for i, j in data.head(10000).iterrows():
        image_name, confidence = X[i], y[i]
        image_id = image_name.split('.')[0]
        filename = 'out' + '/' + image_name
        image = load_img(filename, target_size=(50,50))
        image = img_to_array(image)
        X_data.append(image)
        Y_data.append(confidence)
    X_data, Y_data = array(X_data), array(Y_data)
    return X_data, Y_data


# In[ ]:


a, b = create_data()


# In[1]:


#Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(a, b, test_size=0.2)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

class_num = y_test.shape[1]

print("class_num", class_num)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

del a


# In[4]:


### Creating Model
model = Sequential()

#model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))
model.add(Conv2D(32, (3, 3), input_shape=(50, 50, 3), activation='relu', padding='same'))
model.add(Activation('relu'))

#model.add(Conv2D(224, (3, 3), input_shape=(3, 224, 224), activation='relu', padding='same'))

model.add(Dropout(0.2))

model.add(BatchNormalization())

#2nd convolutional layer
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

#3rd convolutional layer
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(256, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
    
model.add(Dense(128, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

#final layer should have as many neurons as there are classes. Since we have one class and it is regression problem so the value of final neuron will be between 0-99
model.add(Dense(class_num))
model.add(Activation('softmax'))

epochs = 25
optimizer = 'adam'

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# In[5]:


print(model.summary())


# In[6]:


numpy.random.seed(21)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)


# In[ ]:


# Model evaluation
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

