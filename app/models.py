from datetime import datetime
from flask_login import UserMixin
from app import db

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    uploads = db.relationship('UploadHistory', backref='user', lazy=True)  # Relationship to UploadHistory

    def __repr__(self):
        return f"User('{self.username}', '{self.email}')"

class UploadHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  # Foreign key to User
    image_path = db.Column(db.String(200), nullable=False)  # Path to the uploaded image
    predicted_class = db.Column(db.String(50), nullable=False)  # Predicted class
    upload_time = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)  # Timestamp of upload

    def __repr__(self):
        return f"UploadHistory('{self.image_path}', '{self.predicted_class}', '{self.upload_time}')"