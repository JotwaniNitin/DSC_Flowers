
# coding: utf-8

# In[1]:


import sys
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks


# In[2]:


DEV = False
argvs = sys.argv
argc = len(argvs)


# In[3]:


if argc > 1 and (argvs[1] == "--development" or argvs[1] == "-d"):
  DEV = True

if DEV:
  epochs = 2
else:
  epochs = 10


# In[4]:


train_data_path = '/home/nitin/Desktop/flowers2'
validation_data_path = '/home/nitin/Desktop/duplicate2'


# In[5]:


"""
Parameters
"""
img_width, img_height = 150, 150
batch_size = 32
samples_per_epoch = 1000
validation_steps = 300
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 2
pool_size = 2
classes_num = 5
lr = 0.0004


# In[6]:


model = Sequential()
model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, border_mode ="same", input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))


# In[7]:


model.add(Convolution2D(nb_filters2, conv2_size, conv2_size, border_mode ="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size), dim_ordering='th'))


# In[8]:


model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(classes_num, activation='softmax'))


# In[9]:


model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])


# In[10]:


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


# In[11]:


test_datagen = ImageDataGenerator(rescale=1. / 255)


# In[12]:


train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')


# In[13]:


validation_generator = test_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')


# In[14]:


"""
Tensorboard log
"""
log_dir = './tf-log/'
tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
cbks = [tb_cb]


# In[17]:


model.fit_generator(
    train_generator,
    samples_per_epoch=samples_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=cbks,
    validation_steps=validation_steps)


# In[19]:


target_dir = './models/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
model.save('./models/model.h5')
model.save_weights('./models/weights.h5')


# In[20]:


import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model


# In[21]:


img_width, img_height = 150, 150
model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)


# In[22]:


def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    print("Label: Daisy")
  elif answer == 1:
    print("Label: Dandelion")
  elif answer == 2:
    print("Label: Rose")
  elif answer == 3:
    print("Label: Sunflower")
  elif answer == 4:
    print("Label: Tulip")

  return answer

daisy_t = 0
daisy_f = 0
rose_t = 0
rose_f = 0
sunflower_t = 0
sunflower_f = 0
dandelion_t = 0
dandelion_f = 0
tulip_t = 0
tulip_f = 0


# In[23]:


for i, ret in enumerate(os.walk('/home/nitin/Desktop/flowers2/daisy')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    #print("Label: Daisy")
    result = predict(ret[0] + '/' + filename)
    if result == 0:
      daisy_t += 1
    else:
      daisy_f += 1


# In[24]:


for i, ret in enumerate(os.walk('/home/nitin/Desktop/flowers2/dandelion')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    #print("Label: dandelion")
    result = predict(ret[0] + '/' + filename)
    if result == 2:
      dandelion_t += 1
    else:
      dandelion_f += 1


# In[25]:


for i, ret in enumerate(os.walk('/home/nitin/Desktop/flowers2/rose')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    #print("Label: Rose")
    result = predict(ret[0] + '/' + filename)
    if result == 1:
      rose_t += 1
    else:
      rose_f += 1


# In[26]:


for i, ret in enumerate(os.walk('/home/nitin/Desktop/flowers2/sunflower')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    #print("Label: Sunflower")
    result = predict(ret[0] + '/' + filename)
    if result == 2:
      sunflower_t += 1
    else:
      sunflower_f += 1


# In[27]:


for i, ret in enumerate(os.walk('/home/nitin/Desktop/flowers2/tulip')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    #print("Label: Tulip")
    result = predict(ret[0] + '/' + filename)
    if result == 2:
      tulip_t += 1
    else:
      tulip_f += 1


# In[28]:


"""
Check metrics
"""
print("True Daisy: ", daisy_t)
print("False Daisy: ", daisy_f)
print("True Rose: ", rose_t)
print("False Rose: ", rose_f)
print("True Sunflower: ", sunflower_t)
print("False Sunflower: ", sunflower_f)
print("True Dandelion: ", dandelion_t)
print("False Dandelion: ", dandelion_f)
print("True Tulip: ", tulip_t)
print("False Tulip: ", tulip_f)

