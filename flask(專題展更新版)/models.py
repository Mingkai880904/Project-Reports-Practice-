
from db import db

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Text, nullable=False)
    user_name= db.Column(db.Text, nullable=False)
    email = db.Column(db.Text, nullable=False)
    password = db.Column(db.Text, nullable=False)

class Img(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    img = db.Column(db.LargeBinary)
    name = db.Column(db.Text, nullable=False)
    pa_name= db.Column(db.Text, nullable=False)
    datetime = db.Column(db.Text, nullable=False)
    mimetype = db.Column(db.Text, nullable=False)
    answer =db.Column(db.Text)