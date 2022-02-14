import os
import secrets
import numpy as np

from PIL import Image
from fileinput import filename
from turtle import title
from flask import render_template, url_for, flash, redirect,request,abort
from flaskblog import app,db,bcrypt #從flaskblog資料夾 import app db bcrypt
from flaskblog.forms import RegistrationForm, LoginForm,UpdateAccountForm,PostForm
from flaskblog.models import User,Post
from flask_login import login_user,current_user,logout_user,login_required
from werkzeug.utils import secure_filename

from tensorflow.python.keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image


import base64

MODEL_PATH = 'flaskblog/models/trained_1128.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()





@app.route("/", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:#如果使用者已經成功登入
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user=User.query.filter_by(email = form.email.data).first()
        if user and bcrypt.check_password_hash(user.password,form.password.data): #判斷使用者是否存在跟判斷資料表的密碼與表單輸入的是否相同
            login_user(user,remember=form.remember.data)
            next_page = request.args.get('next')#有下一頁
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html', title='Login', form=form)


@app.route("/home")
def home():
    posts = Post.query.all()
    return render_template('home.html', posts=posts)


@app.route("/about")
def about():
    return render_template('aboutproject.html', title='About')

@app.route("/update2")
def update2():
    return render_template('update.html', title='Update2')


def model_predict(img_path, model):
    # img = image.load_img(img_path, target_size=(64, 64)) #trained_model.h5
    img = image.load_img(img_path, target_size=(300, 300))  # trained_1114.h5
    # Preprocessing the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)

    return preds



@app.route('/upload', methods=['GET', 'POST'])
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
        print(preds)
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

        # # points to the beginning of the file stream
        # pic.stream.seek(0)  # 之前的錯誤發生，是pic檔案讀取(以stream格式)後，指標並不在最開始的位置
        # # 所以必須將pic.stream調整回最起始的位置seek(0)才能讀到原本的圖片檔

        # # 準備要上傳到資料庫的圖片資料
        # mimetype = pic.mimetype
        # name = request.form['patient_name']
        # pa_name = request.form['pa_name']
        # datetime = request.form['datetime']
        # # print(name)
        # # print(datetime)
        # # print('pic --> ', pic)
        # img_to_db = Img(img=pic.read(), name=name,pa_name=pa_name,
        #                 mimetype=mimetype, datetime=datetime, answer=result)  # 準備Img 類別需要的參數 (定義在model.py)
        # db.session.add(img_to_db)  # 新增圖片到資料庫db中
        # db.session.commit()  # 執行

        # # removes file from the server after prediction has been returned
        # os.remove(file_path)  # 將圖片檔從我們暫存的路徑中移除 (圖片已經上傳到資料庫db中)
        return result
    return render_template('upload.html', title='Upload')




@app.route("/query")
def query():
    return render_template('query.html', title='Query')


@app.route("/delete")
def delete():
    return render_template('delete.html', title='Delete')


@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:#如果使用者已經成功登入
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8') #將使用者輸入密碼加密
        user = User(username=form.username.data,email=form.email.data,password=hashed_password) #使用者名字email密碼(密碼使用hash)是從表單取得
        db.session.add(user)
        db.session.commit()
        flash('你的帳號已建立，請重新登入', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)




@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))



def save_picture(form_picture):#用來儲存圖檔
    random_hex = secrets.token_hex(8)
    _,f_ext = os.path.splitext(form_picture.filename)#將檔名 附檔名做分割存在前面2個變數
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path,'static/profile_pics',picture_fn)
    
    output_size = (125,125) #設定圖片尺寸(縮小到125*125)
    i =Image.open(form_picture)
    i.thumbnail(output_size)#縮圖

    i.save(picture_path)
    
    return picture_fn


@app.route("/account", methods=['GET', 'POST'])
@login_required
def account():
    form = UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
            current_user.image_file = picture_file
        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        flash('你的帳號已經更新','success')
        return redirect(url_for('account'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email    
    image_file = url_for('static',filename='profile_pics/'+current_user.image_file)
    return render_template('account.html',title='Account',image_file=image_file,form=form)    



@app.route("/post/new", methods=['GET', 'POST'])
@login_required
def new_post():
    form = PostForm()
    if form.validate_on_submit():
        post = Post(title = form.title.data,content = form.content.data,author=current_user)
        db.session.add(post)
        db.session.commit()
        flash('你的貼文已經建立!','success')
        return redirect(url_for('home'))
    return render_template('create_post.html',title='New Post',form=form,legend='New Post')


@app.route("/post/<post_id>")
def post(post_id):
    post =Post.query.get_or_404(post_id) #有get到值，沒有則顯示404
    return render_template('post.html',title=post.title,post=post)    



@app.route("/post/<post_id>/update", methods=['GET', 'POST'])
@login_required
def update_post(post_id):
    post =Post.query.get_or_404(post_id) #有get到值，沒有則顯示404
    if post.author != current_user: #若post作者不是登入的人不能做更新 
        abort(403)
    form = PostForm()    
    if form.validate_on_submit():
        post.title = form.title.data
        post.content = form.content.data
        db.session.commit()
        flash('你的貼文已更新!','success')
        return redirect(url_for('post',post_id=post.id))
    elif request.method == 'GET':    
        form.title.data = post.title
        form.content.data = post.content
    return render_template('create_post.html',title='Update Post',form=form,legend='Update Post')    

   

@app.route("/post/<post_id>/delete", methods=['POST'])
@login_required
def delete_post(post_id):   
    post =Post.query.get_or_404(post_id) #有get到值，沒有則顯示404
    if post.author != current_user: #若post作者不是登入的人不能做更新 
        abort(403)
    db.session.delete(post)
    db.session.commit()    
    flash('你的貼文已刪除!','success')
    return redirect(url_for('home'))
