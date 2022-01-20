import os
import secrets
from PIL import Image
from fileinput import filename
from turtle import title
from flask import render_template, url_for, flash, redirect,request
from flaskblog import app,db,bcrypt #從flaskblog資料夾 import app db bcrypt
from flaskblog.forms import RegistrationForm, LoginForm,UpdateAccountForm
from flaskblog.models import User,Post
from flask_login import login_user,current_user,logout_user,login_required

posts = [
    {
        'author': 'Corey Schafer',
        'title': 'Blog Post 1',
        'content': 'First post content',
        'date_posted': 'April 20, 2018'
    },
    {
        'author': 'Jane Doe',
        'title': 'Blog Post 2',
        'content': 'Second post content',
        'date_posted': 'April 21, 2018'
    }
]


@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html', posts=posts)


@app.route("/about")
def about():
    return render_template('about.html', title='About')


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


@app.route("/login", methods=['GET', 'POST'])
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