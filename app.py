from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms, models
from PIL import Image
from datetime import datetime
import os
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB limit
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load and optimize model
def load_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 4)
    try:
        model.load_state_dict(
            torch.load('maize_resnet18.pth', 
                      map_location=torch.device('cpu'))
        )
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")

model = load_model()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Database Models
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False, index=True)
    password = db.Column(db.String(120), nullable=False)
    scans = db.relationship('Scan', backref='user', lazy='dynamic')

class Scan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), index=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def validate_image(file_stream):
    try:
        Image.open(file_stream).verify()
        file_stream.seek(0)
        return True
    except Exception:
        return False

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        app.logger.info(f"{func.__name__} executed in {end-start:.2f}s")
        return result
    return wrapper

@app.route('/')
def home():
    return render_template('home.html', active_page='home')

@app.route('/about')
def about():
    return render_template('about.html', active_page='about')

@app.route('/reports', methods=['GET', 'POST'])
@login_required
@timeit
def reports():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded', 'danger')
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)

        try:
            if not validate_image(file.stream):
                flash('Invalid image format (PNG/JPG/JPEG only)', 'danger')
                return redirect(request.url)

            filename = secure_filename(f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}")
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Process image directly from memory
            img = Image.open(file.stream).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)
            
            # Verify tensor dimensions
            if img_tensor.shape != torch.Size([1, 3, 256, 256]):
                raise ValueError("Invalid tensor dimensions after transformation")

            # Make prediction
            with torch.no_grad():
                outputs = model(img_tensor)
            
            _, predicted = torch.max(outputs, 1)
            classes = ['Healthy', 'Grey-leaf-spot', 'Common-rust', 'Blight']
            result = classes[predicted.item()]

            # Save image and database record
            img.save(file_path)
            new_scan = Scan(
                filename=filename,
                prediction=result,
                user_id=current_user.id
            )
            db.session.add(new_scan)
            db.session.commit()

            flash(f'Prediction: {result}', 'success')
            return redirect(url_for('reports'))

        except Exception as e:
            db.session.rollback()
            flash(f'Error processing image: {str(e)}', 'danger')
            app.logger.error(f"Prediction error: {str(e)}")
            return redirect(request.url)

    scans = current_user.scans.order_by(Scan.timestamp.desc()).all()
    return render_template('reports.html', 
                         scans=scans,
                         active_page='reports')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('reports'))
        flash('Invalid username or password', 'danger')
    return render_template('login.html', active_page='login')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return redirect(url_for('signup'))
        
        try:
            new_user = User(
                username=username,
                password=generate_password_hash(password)
            )
            db.session.add(new_user)
            db.session.commit()
            flash('Account created successfully! Please login', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('Error creating account. Please try again.', 'danger')
            app.logger.error(f"Signup error: {str(e)}")
    
    return render_template('signup.html', active_page='signup')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=False)