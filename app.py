from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import torch
from torchvision import transforms, models
from PIL import Image
from datetime import datetime
import os
import imghdr

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB limit
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load trained model
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 4)  # 4 classes
model.load_state_dict(torch.load('maize_resnet18.pth', map_location=torch.device('cpu')))
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Database Models
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    scans = db.relationship('Scan', backref='user', lazy=True)

class Scan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/reports', methods=['GET', 'POST'])
@login_required
def reports():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)
        
        if file:
            # Validate image
            if not validate_image(file.stream):
                flash('Invalid image format. Allowed formats: PNG, JPG, JPEG', 'danger')
                return redirect(request.url)
            
            try:
                # Save file
                filename = secure_filename(f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}")
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                # Make prediction
                image = Image.open(file_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0)
                
                with torch.no_grad():
                    outputs = model(image_tensor)
                
                _, predicted = torch.max(outputs, 1)
                classes = ['Healthy', 'Grey-leaf-spot', 'Common-rust', 'Blight']
                result = classes[predicted.item()]

                # Save to database
                new_scan = Scan(
                    filename=filename,
                    prediction=result,
                    user_id=current_user.id
                )
                db.session.add(new_scan)
                db.session.commit()

                flash(f'Prediction: {result}', 'success')

            except Exception as e:
                db.session.rollback()
                flash('Error processing image. Please try again.', 'danger')
                print(f"Error: {str(e)}")

            return redirect(url_for('reports'))
    
    # Get user's scan history
    scans = Scan.query.filter_by(user_id=current_user.id).order_by(Scan.timestamp.desc()).all()
    return render_template('reports.html', scans=scans)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('reports'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

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
        
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password=hashed_password)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Account created successfully! Please login', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('Error creating account. Please try again.', 'danger')
            print(f"Error: {str(e)}")
    
    return render_template('signup.html')

def validate_image(stream):
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None, header)
    return format in {'png', 'jpeg', 'jpg'}

def secure_filename(filename):
    return filename.replace(" ", "_").lower()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)