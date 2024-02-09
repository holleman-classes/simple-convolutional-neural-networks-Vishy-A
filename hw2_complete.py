### Add lines to import modules as needed
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense
## 

def build_model1():

  model = Sequential([
  
    Conv2D(32,  (3, 3), strides = (2, 2), padding = 'same', activation = 'relu', input_shape = (32, 32, 3)), 
    BatchNormalization(),
    Conv2D(64,  (3, 3), strides = (2, 2), padding = 'same', activation = 'relu'),
    BatchNormalization(),
    Conv2D(128,  (3, 3), strides = (2, 2), padding  = 'same', activation = 'relu'),
    BatchNormalization(),
    Conv2D(128,  (3, 3), padding = 'same', activation = 'relu'),
    BatchNormalization(),
    Conv2D(128,  (3, 3), padding = 'same', activation = 'relu'),
    BatchNormalization(),
    Conv2D(128,  (3, 3), padding = 'same', activation = 'relu'),
    BatchNormalization(),
    Conv2D(128,  (3, 3), padding = 'same', activation = 'relu'),
    BatchNormalization(),
    MaxPooling2D( (4, 4), strides = (4, 4)),
    Flatten(),
    Dense(128, activation = 'relu'),
    BatchNormalization(),
    Dense(10, activation = 'softmax')

  ])
  return model

def build_model2():
  
  model = Sequential([
  
    Conv2D(32,  (3, 3), strides = (2, 2), padding = 'same', activation = 'relu', input_shape = (32, 32, 3)), 
    BatchNormalization(),
    DepthwiseConv2D((3, 3), strides = (2, 2), padding = 'same', activation = 'relu', use_bias = False),
    BatchNormalization(),
    DepthwiseConv2D((3, 3), strides = (2, 2), padding  = 'same', activation = 'relu',use_bias = False),
    BatchNormalization(),
    DepthwiseConv2D((3, 3), padding = 'same', activation = 'relu',use_bias = False),
    BatchNormalization(),
    DepthwiseConv2D((3, 3), padding = 'same', activation = 'relu',use_bias = False),
    BatchNormalization(),
    DepthwiseConv2D((3, 3), padding = 'same', activation = 'relu',use_bias = False),
    BatchNormalization(),
    DepthwiseConv2D((3, 3), padding = 'same', activation = 'relu',use_bias = False),
    BatchNormalization(),
    MaxPooling2D( (4, 4), strides = (4, 4)),
    Flatten(),
    Dense(128, activation = 'relu'),
    BatchNormalization(),
    Dense(10, activation = 'softmax')

  ])  
  return model

def build_model3():
  inputs = layers.Input(shape = (32, 32, 3))
  x = inputs

  x = layers.Conv2D(32, (3, 3), strides = (2, 2), padding = 'same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation('relu')(x)
  x = layers.Dropout(.2)(x)

  for _ in range(3):
    shortcut = x

    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.add([x, shortcut])

  x = layers.GlobalAveragePooling2D()(x)
    
  x = layers.Dense(128, activation='relu')(x)
  x = layers.Dropout(0.2)(x)
    
  outputs = layers.Dense(10, activation='softmax')(x)
  
  model = models.Model(inputs, outputs)

  return model

def build_model50k():
  model = None # Add code to define model 1.
  return model

# no training or dataset construction should happen above this line
if __name__ == '__main__':

  ########################################
  ## Add code here to Load the CIFAR10 data set
  (train_images_full, train_labels_full), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
  train_images, val_images, train_labels, val_labels = train_test_split(train_images_full, train_labels_full, test_size=0.2, random_state=42)
  
  ########################################
  ## Build and train model 1
  model1 = build_model1()
  # compile and train model 1.
  model1.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
  model1.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels))
  ## Build, compile, and train model 2 (DS Convolutions)
  model2 = build_model2()
  model2.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
  model2.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels))
  
  ### Repeat for model 3 and your best sub-50k params model
  model3 = build_model3()
  model3.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
  model3.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels))
  