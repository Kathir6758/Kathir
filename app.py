import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from tensorflow.keras.models import load_model
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from werkzeug.utils import secure_filename

# Initialize Flask app

app = Flask(__name__, instance_path=os.path.abspath('instance'))

app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# Load the CNN model
model = load_model('CNN.model')

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

# User Authentication Model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Preprocess Image
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.medianBlur(img, 1)
    img = cv2.resize(img, (50, 50))
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

# Predict Image
def predict_image(image_path):
    img_processed = preprocess_image(image_path)
    prediction = model.predict(img_processed)
    return np.argmax(prediction)

# Generate AI Explanation
def generate_gpt2_response(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = gpt2_model.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
    return tokenizer.decode(output[0], skip_special_tokens=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        user = User(username=username, password=password)
        db.session.add(user)
        db.session.commit()
        flash("Account created successfully! You can now log in.", "success")
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and bcrypt.check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash("Login failed. Check username and password.", "danger")
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Predict and generate explanation
            predicted_class = predict_image(filepath)
            conditions = ["Benign", "Early Stage", "Pre Acute Stage", "Pro Acute Stage"]
            explanation = generate_gpt2_response(f"Acute Lymphoblastic Leukemia detected as {conditions[predicted_class]}. Explain this condition.")
            
            return render_template('dashboard.html', user=current_user, image=filename, prediction=conditions[predicted_class], explanation=explanation)

    return render_template('dashboard.html', user=current_user)

if __name__ == "__main__":
    db.create_all()
    
    app.run(debug=True)
