from __future__ import division, print_function
from re import A
from flask import Flask, render_template, request,flash
import numpy as np
import os
import cv2
import time
from datetime import datetime
from numpy.core.records import array

from tensorflow.python.keras.preprocessing.image import img_to_array


from db import db_init, db
from models import Img

from keras.preprocessing.image import load_img  # test
from PIL import Image
# from PIL import Image, ImageOps  # test
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

from werkzeug.utils import secure_filename


import base64

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///img1227.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'cnn'

db_init(app)


# Model saved with Keras model.save()
# MODEL_PATH = 'models/pneumonia_cnn1011.h5'
# MODEL_PATH = 'models/trained_model.h5'
MODEL_PATH = 'models/trained_1128.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()


@app.route('/')
def hello_world():
    return render_template("index.html")
   
database = {'admin': '123', 'ksu': 'ksu' , 'cnn': 'cnn'}


@app.route('/home', methods=['POST', 'GET'])
def home():
    return render_template('upload.html')


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
    img = image.load_img(img_path, target_size=(300, 300))  # trained_1114.h5
    # Preprocessing the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)

    return preds


# 預測及上傳圖片到資料庫
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        # Get the file from post request
        pic = request.files['pic']  # 只能使用一次

        #  Save the file to ./uploads
        basepath = os.path.dirname(__file__)  # 取得伺服器上的路徑與目錄
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(pic.filename))
        pic.save(file_path)
        # Make prediction
        preds = model_predict(file_path, model)
        # print('preds = ', preds)
        # print('np.max(preds) = ', np.max(preds))
        x = str(np.max(preds))

        # Arrange the correct return according to the model.
        # In this model 1 is Pneumonia and 0 is Normal.
        str1 = 'Pneumonia (肺炎)'  # 不要列印x，因為會誤會是辨識率 (x是label)
        str2 = 'Normal (正常)'

        if preds > 0.5:
            result = str1
        else:
            result = str2

        # print('predict result --->', result)

        # points to the beginning of the file stream
        pic.stream.seek(0)  # 之前的錯誤發生，是pic檔案讀取(以stream格式)後，指標並不在最開始的位置
        # 所以必須將pic.stream調整回最起始的位置seek(0)才能讀到原本的圖片檔

        # 準備要上傳到資料庫的圖片資料
        mimetype = pic.mimetype
        name = request.form['patient_name']
        pa_name = request.form['pa_name']
        datetime = request.form['datetime']
        # print(name)
        # print(datetime)
        # print('pic --> ', pic)
        img_to_db = Img(img=pic.read(), name=name, pa_name=pa_name,
                        mimetype=mimetype, datetime=datetime, answer=result)  # 準備Img 類別需要的參數 (定義在model.py)
        db.session.add(img_to_db)  # 新增圖片到資料庫db中
        db.session.commit()  # 執行

        # removes file from the server after prediction has been returned
        os.remove(file_path)  # 將圖片檔從我們暫存的路徑中移除 (圖片已經上傳到資料庫db中)
    return result


'''
@app.route("/query", methods=["GET", "POST"])
def query():
    if request.method == "POST":
        name = request.form['patient_name']
        img = Img.query.filter_by(name=name).first()
           # img = Img.query.filter(name==name).all()

           # 有這位病患
        try:
            image = img.img
                # print(img.img)
        except:
                # 病患不存在 傳回錯誤
            image = None
            return '查詢不到此資料'
            # 讀取byte64結構，存成圖片並傳入地址
        nparr = np.fromstring(img.img, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
        now = datetime.now().timestamp()
        cv2.imwrite('./static/displayDB/temp'+str(now)+'.png', image)
        time.sleep(2)
        # print(image='temp.png', name=img.name, datetime=img.datetime)
        return render_template("query.html", image='temp'+str(now)+'.png', name=img.name, datetime=img.datetime)
    return render_template("query.html", image=None)
'''


@app.route("/query", methods=["GET", "POST"])
def query():
    if request.method == "POST":
        input = request.form['patient_name']
        all_img = Img.query.filter(Img.name == input).all()
        #decode_all_img = []
        print(all_img)
        # if all_img == []:
        #     return '查詢不到該筆紀錄'   #當all_img是[]，顯示查詢不到
        # for img in all_img:
        for i in range(len(all_img)):
            # nparr = np.fromstring(img.img, np.uint8)# 讀取byte64結構，存成圖片並傳入地址
            #image_a = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
            #now = datetime.now().timestamp()
            #cv2.imwrite('./static/displayDB/temp'+str(now)+'.png', image_a)
            # time.sleep(2)
            # image_a='temp'+str(now)+'.png'

            # print(all_img[i].answer)
            all_img[i].img = base64.b64encode(all_img[i].img).decode('ascii')
            # print(all_img[i].img)

        return render_template("query.html", to_quest=all_img)
    return render_template("query.html", image=None)

    # return render_template("query.html")
    # nparr = np.fromstring(img.img, np.uint8)
    # image = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    # now = datetime.now().timestamp()
    # cv2.imwrite('./static/displayDB/temp'+str(now)+'.png', image)

    # try:
    #     image = img.img
    #         # print(img.img)
    # except:
    #         # 病患不存在 傳回錯誤
    #     image = None
    #     return '查詢不到此資料'

    # nparr = np.fromstring(img.img, np.uint8)
    # image = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    # now = datetime.now().timestamp()
    # cv2.imwrite('./static/displayDB/temp'+str(now)+'.png', image)
    # time.sleep(2)
    # # print(image='temp.png', name=img.name, datetime=img.datetime)
    # return render_template("query.html", image='temp'+str(now)+'.png', name=img.name, datetime=img.datetime,result=img.answer)

    # return render_template("query.html")


@app.route("/delete", methods=["GET", "POST"])
def delete():
    if request.method == "POST":

        id = request.form['patient_name']
        img = Img.query.filter_by(id=id).first()
        if img == None:
            # return '未查詢到欲刪除資料!'
            str4 = 'Sorry,查詢不到該筆紀錄'#當沒有搜尋到指定id，顯示str4文字(查詢不到)
            return render_template("delete.html",no_data_id=str4)
        db.session.delete(img)
        db.session.commit()
        flash('該筆紀錄已刪除!!', 'success')
    return render_template("delete.html")
# 先查詢，再刪除


'''
@app.route("/update", methods=["GET", "POST"])
def update():
    if request.method == "POST":

        name = request.form['patient_name']
        update_name = request.form['update_name']
        img = Img.query.filter_by(name=name).first()
        img.name = update_name
        db.session.commit()
    return render_template("update.html")
# 先查詢，再更新
'''


@app.route("/update", methods=["GET", "POST"])
def update():
    if request.method == "POST":

        search_id = request.form['patient_name']
        update_name = request.form['update_name']
        update_datetime = request.form['update_datetime']
        img = Img.query.filter_by(id=search_id).first()
        if img == None:
            str4 = 'Sorry,查詢不到該筆紀錄'#當沒有搜尋到指定id，顯示str4文字(查詢不到)
            return render_template("update.html",no_data=str4)
        img.pa_name = update_name
        img.datetime = update_datetime
        db.session.commit()
        flash('此紀錄已更新!!', 'success')
    return render_template("update.html")
# 先查詢，再更新


if __name__ == '__main__':
    app.run(debug=True)
