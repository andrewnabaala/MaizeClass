from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from app.models import User, UploadHistory
from app import db
import os
from app.predict import load_model, predict
from datetime import datetime

# Load the trained model
model = load_model('app/models/maize_resnet18.pth')

# Create a Blueprint for the main routes
main = Blueprint('main', __name__)

# Home Page
@main.route('/', methods=['GET', 'POST'])
@login_required
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            upload_folder = 'app/static/uploads'
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)

            # Make a prediction
            predicted_class = predict(file_path, model)

            # Save the upload history
            new_upload = UploadHistory(
                user_id=current_user.id,
                image_path=file_path,
                predicted_class=predicted_class,
                upload_time=datetime.utcnow()
            )
            db.session.add(new_upload)
            db.session.commit()

            # Render the result with the correct image URL
            image_url = url_for('static', filename=f'uploads/{file.filename}')
            return render_template('index.html', predicted_class=predicted_class, image_url=image_url)

    return render_template('index.html')

# Reports Page
@main.route('/reports')
@login_required
def reports():
    # Fetch the user's upload history
    upload_history = UploadHistory.query.filter_by(user_id=current_user.id).order_by(UploadHistory.upload_time.desc()).all()
    return render_template('reports.html', upload_history=upload_history)

# Login Route
@main.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and user.password == password:  # Plain text password comparison
            login_user(user)
            return redirect(url_for('main.index'))
        else:
            flash('Invalid email or password.', 'error')
    return render_template('login.html')

# Register Route
@main.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        # Check if user already exists
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'error')
            return redirect(url_for('main.register'))

        new_user = User(username=username, email=email, password=password)  # Save plain text password
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('main.login'))
    return render_template('register.html')

# Logout Route
@main.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.index'))

# About Page
@main.route('/about')
def about():
    return render_template('about.html')