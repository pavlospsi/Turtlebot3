import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as npimg
import os
from numpy.core.defchararray import asarray
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint

import cv2
import pandas as pd
import random
import ntpath


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.numpy_ops import np_arrays


datadir = 'Self-Driving-Car'
columns = ['PATH','SPEED','ANGULAR']
data = pd.read_csv(os.path.join(datadir,'driving_log.csv'),names = columns)
pd.set_option('display.max_colwidth', None)
data.head()


num_bins = 25
samples_per_bin = 10
hist, bins = np.histogram(data['ANGULAR'], num_bins)
center = bins[:-1] + bins[1:]*0.05 

## Plot
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['ANGULAR']), np.max(data['ANGULAR'])), (samples_per_bin, samples_per_bin))
plt.show()
print('Total data: {0}'.format(len(data)))

def load_img_steering(datadir, data):
  """Get img and steering data into arrays"""
  image_path = []
  steering = []
  for i in range(len(data)):
    indexed_data = data.iloc[i]
    center= indexed_data[0]
    image_path.append(os.path.join(datadir, center))
    steering.append(float(indexed_data[2]))
    image_paths = np.asarray(image_path)
    steerings = np.asarray(steering)
  return image_paths, steerings

image_paths, steerings = load_img_steering(datadir + '/IMAGES', data)
X_train, X_valid, Y_train, Y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=5)
print(len(X_train),len(X_valid))

print("Training Samples1: {}\nTraining Samples2: {}\nValid Samples1: {}\nValid Samples2: {}".format(len(X_train),len(Y_train),len(X_valid),len(Y_valid)))
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(Y_train, bins=num_bins, width=0.05, color='blue')
axes[0].set_title('Training set')
axes[1].hist(Y_valid, bins=num_bins, width=0.05, color='red')
axes[1].set_title('Validation set')
plt.show()

def img_preprocess(img):

  #img = cv2.GaussianBlur(img, (3, 3), 0)
  img = cv2.resize(img, (159, 119))
  img=img[:,:,::-1]
  img = img /255
  return img



  

def batch_generator(image_paths,steerings,batch_size):
  while True:
    imageBatch = []
    steeringBatch = []
    for i in range (batch_size):
      index=random.randint(0,len(image_paths)-1)
      if os.path.isfile(image_paths[index]):
        img=npimg.imread(image_paths[index])
        steering=steerings[index]
    
        img=img_preprocess(img)
        imageBatch.append(img)
        steeringBatch.append(steering)
      i=i+1
      
    yield (np.asarray(imageBatch),np.asarray(steeringBatch))
      

def image_data_generator(image_paths, steering_angles, batch_size, is_training):
    while True:
        batch_images = []
        batch_steering_angles = []
        
        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)
            if os.path.isfile(image_paths[random_index]):
                image_path = image_paths[random_index]
                image = npimg.imread(image_paths[random_index])
                steering_angle = steering_angles[random_index]
              
                image = img_preprocess(image)
                batch_images.append(image)
                batch_steering_angles.append(steering_angle)
            
                yield( np.asarray(batch_images), np.asarray(batch_steering_angles))

def nvidia_model():
    model = Sequential(name='Nvidia_Model')
    model.add(Conv2D(24, (5, 5), strides=(2, 2), input_shape=(119,159, 3), activation='elu')) 
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu')) 
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu')) 
    model.add(Conv2D(64, (3, 3), activation='elu')) 
    model.add(Dropout(0.2)) 
    model.add(Conv2D(64, (3, 3), activation='elu')) 

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    

    model.add(Dense(1)) 

    optimizer = Adam(lr=1e-3)
    model.compile(loss='mse', optimizer=optimizer)
    
    return model

model = nvidia_model()
print(model.summary())
filepath = "../../weights/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint( filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1) 

history = model.fit_generator(image_data_generator( X_train, Y_train, batch_size=100, is_training=True),
                              steps_per_epoch=300,
                              epochs=10,
                              validation_data = image_data_generator( X_valid, Y_valid, batch_size=100, is_training=False),
                              validation_steps=200,
                              verbose=1,
                              shuffle=1,
                              callbacks=[checkpoint])
model.save('final_weights.h5')
#history = model.fit(batch_generator(X_train,Y_train,128),steps_per_epoch=13,batch_size=128, epochs=350, validation_data=batch_generator(X_valid,Y_valid,50),validation_steps=50,verbose=1,callbacks=callbacks_list)
#model.save('model_train_v30_final.h5')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()

from sklearn.metrics import mean_squared_error, r2_score

def summarize_prediction(Y_true, Y_pred):
    mse = mean_squared_error(Y_true, Y_pred)
    r_squared = r2_score(Y_true, Y_pred)
    print(f'mse       = {mse:.2}')
    print(f'r_squared = {r_squared:.2%}')
    
def predict_and_summarize(X, Y):
    model = tf.keras.models.load_model('lane_navigation_final.h5')
    Y_pred = model.predict(X)
    summarize_prediction(Y, Y_pred)
    return Y_pred
  
n_tests = 100
X_test, y_test = next(image_data_generator(X_valid, Y_valid, n_tests, False))
y_pred = predict_and_summarize(X_test, y_test)

#(x,y)= batch_generator2(X_valid,Y_valid,100)

scores = model.evaluate(batch_generator(X_valid,Y_valid,64),batch_size=64, steps=200,verbose=1)
