from flask import *  
import os
from scipy import *
import re
from PIL import Image 
import tensorflow.keras as keras 
import sys 
import base64
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.image import load_img 
from imageio import *
import tensorflow.keras as keras
import numpy as np
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
app = Flask(__name__)  


UPLOAD_FOLDER = "Save_images"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'This is your secret key to utilize session'
@app.route('/')  
def upload():  
    return render_template("Upload.html")  


classes = ['burger', 'butter_naan', 'chai', 'chapati', 'chole_bhature','dal_makhani', 'dhokla', 'fried_rice', 'idli', 'jalebi',
                   'kaathi_rolls', 'kadai_paneer', 'kulfi', 'masala_dosa', 'momos', 'paani_puri', 'pakode', 'pav_bhaji', 'pizza', 'samosa']

@app.route('/success', methods = ['POST','GET'])  
def success():  
    if request.method == 'POST':
        f = request.files['sile']
        img = Image.open(f)
        filename = secure_filename(f.filename)
        print(filename)
        img = Image.open(f.stream)
        with BytesIO() as buf:
            img.save(buf, 'jpeg')
            image_bytes = buf.getvalue()
        encoded_string = base64.b64encode(image_bytes).decode()
        model = tf.keras.models.load_model("my_model/")
        img1 = img.resize((224,224))
        img_array = image.img_to_array(img1)
        img_ = np.expand_dims(img_array, axis=0)
        img_ /= 255.
        prediction = model.predict(img_)
        index = np.argmax(prediction)
        a = str(classes[index])
        return render_template("success.html", name = a,img_data=encoded_string )  





  
if __name__ == '__main__':  
    app.run(host='0.0.0.0',port = '2221')  

