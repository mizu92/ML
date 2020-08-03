from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from uploads.core.models import Document
from uploads.core.forms import DocumentForm
import os, json
from glob import glob


import tensorflow as tf
import keras
import time

from keras_squeezenet import SqueezeNet
from keras.applications import densenet
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.python.keras import backend as K

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
tf.keras.backend.clear_session()
global model
model = SqueezeNet()
#model = keras.applications.densenet.DenseNet121(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
global graph
graph = tf.get_default_graph()

def home(request):
    documents = Document.objects.all()
    return render(request, 'core/home.html', { 'documents': documents })


def simple_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        location = '/home/pi/a/simple-file-upload'+uploaded_file_url
        img = image.load_img('/home/pi/a/simple-file-upload'+uploaded_file_url, target_size=(227, 227))
        img_arr = np.expand_dims(image.img_to_array(img), axis=0)
        toc = time.clock()
        x = preprocess_input(img_arr)
        with graph.as_default():
            preds = model.predict(x)
            result00 = decode_predictions(preds, top=3)[0][0][1]
            result01 = decode_predictions(preds, top=3)[0][0][2]
            result10 = decode_predictions(preds, top=3)[0][1][1]
            result11 = decode_predictions(preds, top=3)[0][1][2]
            result20 = decode_predictions(preds, top=3)[0][2][1]
            result21 = decode_predictions(preds, top=3)[0][2][2]

        tic = time.clock()
        spent = tic - toc
        return render(request, 'core/simple_upload.html', {
             'uploaded_file_url': uploaded_file_url, 'location' : location, 'result00' : result00, 'result01' : result01, 'result10' : result10, 'result11' : result11, 'result20' : result20, 'result21' : result21,'spent' : spent
        })
    return render(request, 'core/simple_upload.html')


def model_form_upload(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('home')
    else:
        form = DocumentForm()
    return render(request, 'core/model_form_upload.html', {
        'form': form
    })
