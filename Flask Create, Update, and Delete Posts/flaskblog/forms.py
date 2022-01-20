from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from flask_login import current_user
from wtforms import StringField, PasswordField, SubmitField, BooleanField,TextAreaField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
from flaskblog.models import User


class RegistrationForm(FlaskForm):
    username = StringField('Username',
                           validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    password = PasswordField('Password',validators=[DataRequired()])
    confirm_passwd = PasswordField('confirm Password',validators=[DataRequired(),EqualTo('password')])
    submit = SubmitField('Sign Up')

    def validate_username(self, username):  # 驗證使用者名稱是否重複
        user = User.query.filter_by(username=username.data).first()
        if user:  # 如果存在
            raise ValidationError('這個名稱已重複,請重新輸入其他名稱')

    def validate_email(self, email):  # 驗證使用者email是否重複
        user = User.query.filter_by(email=email.data).first()
        if user:  # 如果存在
            raise ValidationError('這個email已重複,請重新輸入其他email')


class LoginForm(FlaskForm):
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')


class UpdateAccountForm(FlaskForm):
    username = StringField('Username',
                           validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    picture = FileField('Update Profile Picture', validators=[
                        FileAllowed(['jpg', 'png'])])

    submit = SubmitField('Update')

    def validate_username(self, username):  # 驗證使用者名稱是否重複
        if username.data != current_user.username:  # 如果使用者輸入的名稱不同(確認是否想更新)
            user = User.query.filter_by(username=username.data).first()
            if user:  # 如果存在
                raise ValidationError('這個名稱已重複,請重新輸入其他名稱')

    def validate_email(self, email):  # 驗證使用者email是否重複
        if email.data != current_user.email:  # 如果使用者輸入的email不同(確認是否想更新)
            user = User.query.filter_by(email=email.data).first()
            if user:  # 如果存在
                raise ValidationError('這個email已重複,請重新輸入其他email')


class PostForm(FlaskForm):
    title = StringField('Title',validators=[DataRequired()])
    content = TextAreaField('Content',validators=[DataRequired()])
    submit = SubmitField('Post')