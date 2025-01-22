# 🌽 MaizeGuard - AI-Powered Crop Health Monitor 🛡️

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

![MaizeGuard Demo](https://images.unsplash.com/photo-1601121141203-217d386b1e25?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80)

Revolutionizing agricultural disease detection using deep learning and computer vision.

## 🚀 Features

- 🖼️ Image-based disease detection (4 classes: Healthy, Blight, Grey Leaf Spot, Common Rust)
- 📅 Scan history tracking with visual reports
- 🔐 Secure user authentication system
- 📱 Mobile-responsive interface
- 📈 Performance metrics dashboard
- 🛡️ Security alerts monitoring (CVE tracking)

## 🛠️ Tech Stack

- **AI Engine**: PyTorch 2.3.1 + ResNet-18
- **Web Framework**: Flask 3.0.2
- **Database**: SQLAlchemy 3.1.1 + SQLite
- **Frontend**: Bootstrap 5.3 + Custom CSS

## ⚙️ Installation

1. **Clone Repository**
   ```bash
   git clone https://github.com/yourusername/maizeguard.git
   cd maizeguard
Set Up Virtual Environment

bash
Copy
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
Install Dependencies

bash
Copy
pip install -r requirements.txt
Add AI Model
Place maize_resnet18.pth in project root

Initialize Database

bash
Copy
python -c "from app import db; db.create_all()"
🏃♀️ Running the Application
bash
Copy
python app.py
Visit ➡️ http://localhost:5000

🔒 Security Configuration
python
Copy
# app.py
app.config.update({
    'SECRET_KEY': os.environ.get('SECRET_KEY', 'your-secret-key-here'),
    'MAX_CONTENT_LENGTH': 2 * 1024 * 1024,  # 2MB limit
    'UPLOAD_FOLDER': 'static/uploads'
})
📂 Project Structure
Copy
maizeguard/
├── app.py
├── maize_resnet18.pth
├── requirements.txt
├── static/
│   ├── css/
│   └── uploads/
├── templates/
│   ├── home.html
│   ├── login.html
│   └── ...other templates
└── site.db
🧪 Testing & Security
Run Security Audit

bash
Copy
pip-audit
Check Dependency Vulnerabilities

bash
Copy
pip list --outdated
🌟 Why MaizeGuard?
✅ 95% detection accuracy on test dataset

⏱️ Real-time analysis (<2s per image)

🔄 Automated scan history tracking

📊 Farmer-friendly interface

🚧 Troubleshooting
Common Issues:

🔄 Dependency Conflicts: pip install --force-reinstall -r requirements.txt

🖼️ Image Upload Errors: Check file format (PNG/JPG) and size (<2MB)

🔐 Authentication Issues: Reset database with db.drop_all()

🤝 Contributing
Fork the repository

Create feature branch: git checkout -b feature/amazing-feature

Commit changes: git commit -m 'Add amazing feature'

Push to branch: git push origin feature/amazing-feature

Open Pull Request

📜 License
Distributed under MIT License. See LICENSE for more information.

Need Help?
✉️ Open an issue or contact project maintainers at support@maizeguard.com

Farmers First! 🌱👩🌾
Help us build smarter agriculture solutions