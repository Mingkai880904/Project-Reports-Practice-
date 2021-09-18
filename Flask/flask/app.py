from re import A
from flask import Flask, render_template, request
from h5py._hl import files
from numpy.lib.npyio import load
from tensorflow.python.keras.backend import get_value
app = Flask(__name__)


import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
from keras.preprocessing.image import load_img

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/Sign_up', methods=['GET'])
def signup():
        return render_template('Sign_up.html')


@app.route('/web', methods=['GET'])
def web():
        return render_template('上傳.html')

@app.route('/aboutteam', methods=['GET'])
def team():
        return render_template('aboutteam.html')


@app.route('/aboutproject', methods=['GET'])
def project():
        return render_template('aboutproject.html')


@app.route('/web', methods=['POST'])
def web2():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)  

    model = tensorflow.keras.models.load_model('keras_model.h5', compile=False)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = load_img(image_path,target_size=(224,224))
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    # image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)


    if(prediction[0][0]>prediction[0][1]):
        a=("正常",prediction[0][0])  
        print(a)
    else:
        a = ("肺炎",prediction[0][1])
        print(a)
    response = a
    return render_template('上傳.html',prediction = response)

if __name__ == '__main__':
   app.run(debug = True)