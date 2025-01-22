from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import torch
from torchvision import transforms, models
from PIL import Image
import os
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Ensure the uploads folder exists
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 4)  # 4 classes
model.load_state_dict(torch.load('maize_resnet18.pth', map_location=torch.device('cpu')))
model.eval()

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Scan history (for demonstration purposes)
scan_history = []

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
            # Save the uploaded file
            filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            # Process the image and make a prediction
            image = Image.open(file.stream).convert('RGB')
            image = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = model(image)
            _, predicted = torch.max(output, 1)
            classes = ['Healthy', 'Grey-leaf-spot', 'Common-rust', 'Blight']
            result = classes[predicted.item()]

            # Add the scan to the history
            scan_history.append({
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'image_filename': filename,
                'prediction': result
            })

            flash(f'Prediction: {result}', 'success')
            return redirect(url_for('reports'))
    return render_template('reports.html', scan_history=scan_history, user=current_user)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            login_user(user)
            return redirect(url_for('reports'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User(username=username, password=password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)