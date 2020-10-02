from __future__ import division, print_function
# coding=utf-8
import sys
import os
from datetime import datetime
import glob
import numpy as np
from keras.preprocessing import image 
import random
random.seed(datetime.microsecond)

from keras.applications.imagenet_utils import preprocess_input, decode_predictions

#from keras.models import load_model
import tensorflow as tf
from keras import backend
from tensorflow.keras import backend as K


#tf.logging.set_verbosity(tf.logging.ERROR)

graph=tf.compat.v1.get_default_graph()

#global graph
#graph = tf.get_default_graph()


from skimage.transform import resize

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/major1.h5'

# Load your trained model
#model = load_model(MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
       # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')




@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img = image.load_img(file_path, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        preds = model.predict_classes(x)
        index = ['Mountain Laurel_nonedible','Peppergrass_edible','Purple Deadnettle_edible','rattlebox_nonedible','Rhododendron_nonedible','Toothwort_edible','Wild Grape Vine_edible','Wild Leek_edible']
        text = "prediction : "+ index[preds[0][0]]
        return text
    


if __name__ == '__main__':
    app.run(debug=True,threaded = False)


