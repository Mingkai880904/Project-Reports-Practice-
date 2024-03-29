from __future__ import division, print_function
from re import A
from flask import Flask, render_template, request
import numpy as np
import os
import cv2
import time
from datetime import datetime



from db import db_init, db
from models import Img





# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///img.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db_init(app)







# Model saved with Keras model.save()
# MODEL_PATH = 'models/pneumonia_cnn1011.h5'
# MODEL_PATH = 'models/trained_model.h5'
MODEL_PATH = 'models/trained_1114.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function() 





@app.route('/')
def hello_world():
    return render_template("index.html")


database = {'admin': '123', 'ksu': 'ksu', 'cnn': 'cnn'}


@app.route('/login', methods=['POST', 'GET'])
def login():
    name1 = request.form['username']
    pwd = request.form['password']
    if name1 not in database:
	    return render_template('index.html', info='Invalid User')
    else:
        if database[name1] != pwd:
            return render_template('index.html', info='Invalid Password')
        else:
	         return render_template('upload.html', name=name1)




# @app.route('/')
# def index():
# 	return render_template('index.html')


@app.route('/Sign_up', methods=['GET'])
def signup():
        return render_template('Sign_up.html')


# @app.route('/web', methods=['GET'])
# def web():
#         return render_template('上傳.html')

@app.route('/aboutteam', methods=['GET'])
def team():
        return render_template('aboutteam.html')


@app.route('/aboutproject', methods=['GET'])
def project():
        return render_template('aboutproject.html')


# def model_predict(img_path, model):
#     img = image.load_img(img_path, target_size=(224, 224), grayscale=True)

#     # Preprocessing the image
#     x = image.img_to_array(img)
#     x = np.true_divide(x, 255)
#     x = np.expand_dims(x, axis=0)

#     # Be careful how your trained model deals with the input
#     # otherwise, it won't make correct prediction!
#     # x = preprocess_input(x, mode='caffe')

#     preds = model.predict(x)
#     return preds
def model_predict(img_path, model):
    # img = image.load_img(img_path, target_size=(64, 64)) #trained_model.h5
    img = image.load_img(img_path, target_size=(300, 300)) #trained_1114.h5
    # Preprocessing the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

   
    preds = model.predict(img)
    return preds



# @app.route('/predict', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         # Get the file from post request
#         f = request.files['file']

#         # Save the file to ./uploads
#         basepath = os.path.dirname(__file__)
#         file_path = os.path.join(
#             basepath, 'uploads', secure_filename(f.filename))
#         f.save(file_path)

#         # Make prediction
#         preds = model_predict(file_path, model)

#         # Process your result for human
#         pred_class = preds.argmax(axis=-1)            # Simple argmax
#         # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
#         # result = str(pred_class[0][0][1])               # Convert to string
        
#         # result = str(preds)

#         result = str(np.max(preds))

#         answer=["細菌感染","正常","病毒"]

#         if(result==str(preds[0][0])):
#            print(answer[0])
#            result = answer[0]
#         elif(result==str(preds[0][1])):
#            print(answer[1])
#            result = answer[1]
#         elif(result==str(preds[0][2])):
#            print(answer[2])
#            result = answer[2]
#         return result
#     return None

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        os.remove(file_path)#removes file from the server after prediction has been returned

        # Arrange the correct return according to the model. 
		# In this model 1 is Pneumonia and 0 is Normal.
        str1 = 'Pneumonia，請至胸腔內科治療'
        str2 = 'Normal'
        if preds == 1:
            return str1
        else:
            return str2
    return None



@app.route('/upload', methods=['GET', 'POST'])
def upload2():
	if request.method == "POST":
		pic = request.files['pic']
		if not pic:
			return 'No pic uploaded!', 400
		filename = secure_filename(pic.filename)
		mimetype = pic.mimetype
		if not filename or not mimetype:
			return 'Bad upload!', 400
		#parameter  = request.form['input name']
		name = request.form['patient_name']
		birthday = request.form['birthday']
		print(name)
		print(birthday)
		img = Img(img=pic.read(), name=name, mimetype=mimetype, birthday=birthday)
		db.session.add(img)
		db.session.commit()
	return render_template("upload2.html")



@app.route("/query", methods=["GET", "POST"])
def query():	
	if request.method == "POST":
		
		name = request.form['patient_name']
		img = Img.query.filter_by(name = name).first()
		# img = Img.query.filter(name==name).all()

		#有這位病患
		try:
			image = img.img
			# print(img.img)
		except:
		#病患不存在 傳回錯誤
			image= None
			return '查詢不到此資料'
		#讀取byte64結構，存成圖片並傳入地址
		nparr = np.fromstring(img.img, np.uint8)
		image = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
		now = datetime.now().timestamp()
		cv2.imwrite('./static/displayDB/temp'+str(now)+'.png', image)
		time.sleep(2)
		# print(image='temp.png', name=img.name, birthday=img.birthday)
		return render_template("query.html", image='temp'+str(now)+'.png', name=img.name, birthday=img.birthday)
	return render_template("query.html", image= None )


@app.route("/delete", methods=["GET", "POST"])
def delete():	
	if request.method == "POST":
	
		name = request.form['patient_name']
		img = Img.query.filter_by(name = name).first()
		if img== None:
			return '未查詢到欲刪除資料!'
		db.session.delete(img)
		db.session.commit()
	return render_template("delete.html")
#先查詢，再刪除


@app.route("/update", methods=["GET", "POST"])
def update():	
	if request.method == "POST":
	
		name = request.form['patient_name']
		update_name =request.form['update_name']
		img = Img.query.filter_by(name=name).first()
		img.name = update_name
		db.session.commit()
	return render_template("update.html")
#先查詢，再更新



if __name__ == '__main__':
    app.run(debug=True)
