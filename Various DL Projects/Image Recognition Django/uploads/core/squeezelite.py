
# coding: utf-8

# ### Imports

# In[17]:

#In Jupyter notebooks, you will need to run this command before doing any plotting

import os, json
from glob import glob
import tensorflow as tf
import tensorflow.python.keras
import time


from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.python.keras import backend as K
from keras_squeezenet import *

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np

model = load_model('squeeze.h5')


# In[ ]:

def predict(img_path):
  tic = time.clock()
  img = image.load_img(img_path, target_size=(227, 227))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  print('Input image shape:', x.shape)

  preds = model.predict(x)
  print('Predicted:', decode_predictions(preds))
  toc = time.clock()
  print('Processing time is', toc-tic)


# In[ ]:



