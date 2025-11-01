import os
from datetime import datetime
import json
from pathlib import Path
import joblib

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np


# Import GPU-enabled flask integration
from flask_integration import (
    predict_difficulty, 
    generate_passage, 
    generate_questions,
    extract_features,
    get_system_info,
    EXPERIMENTAL_RESULTS,
    SVM_MODEL,
    GPT2_MODEL,
    DIFFICULTY_MAP_LIST,
    TORCH_AVAILABLE
)

# Try to import torch for GPU detection
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False

from data_pipeline import is_folder_empty_pathlib


app = Flask(__name__)
app.config['SECRET_KEY'] = 'readapt-secret-key-2024'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///readapt.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# ==========================================
# DATABASE MODELS
# ==========================================

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<User {self.email}>'

@login_manager.user_loader
def load_user(user_id):
    """Load user"""
    return db.session.get(User, int(user_id))


# ==========================================
# GPU-ENABLED ML API ROUTES
# ==========================================

@app.route('/api/ml/system-info', methods=['GET'])
def ml_system_info():
    """Get GPU and system information"""
    return jsonify(get_system_info())

@app.route('/api/ml/predict-difficulty', methods=['POST'])
def ml_predict_difficulty():
    """Predict difficulty level of text with GPU acceleration"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    result = predict_difficulty(text)
    return jsonify(result)

@app.route('/api/ml/generate-passage', methods=['POST'])
def ml_generate_passage():
    """Generate adaptive reading passage with GPU acceleration"""
    data = request.get_json()
    topic = data.get('topic', 'general knowledge')
    difficulty = data.get('difficulty', 'intermediate')
    max_length = data.get('max_length', 200)
    
    passage = generate_passage(topic, difficulty, max_length)
    
    return jsonify({
        'passage': passage,
        'topic': topic,
        'difficulty': difficulty,
        'used_gpu': TORCH_AVAILABLE and GPU_AVAILABLE
    })

@app.route('/api/ml/generate-questions', methods=['POST'])
def ml_generate_questions():
    """Generate comprehension questions"""
    data = request.get_json()
    passage = data.get('passage', '')
    num_questions = data.get('num_questions', 3)
    
    if not passage:
        return jsonify({'error': 'No passage provided'}), 400
    
    questions = generate_questions(passage, num_questions)
    
    return jsonify({
        'questions': questions,
        'num_questions': len(questions)
    })

@app.route('/api/ml/analyze-text', methods=['POST'])
def ml_analyze_text():
    """Complete text analysis: difficulty + questions with GPU"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Predict difficulty (GPU-accelerated if available)
    difficulty_result = predict_difficulty(text)
    
    # Generate questions
    questions = generate_questions(text, num_questions=3)
    
    # Extract features
    features = extract_features(text)
    
    return jsonify({
        'text': text[:200] + '...' if len(text) > 200 else text,
        'difficulty': difficulty_result,
        'questions': questions,
        'features': {
            'num_sentences': int(features[0]),
            'num_words': int(features[1]),
            'avg_word_length': float(features[3]),
            'flesch_kincaid_grade': float(features[5])
        },
        'used_gpu': difficulty_result.get('used_gpu', False)
    })

@app.route('/api/ml/experimental-results', methods=['GET'])
def ml_experimental_results():
    """Get experimental results from training"""
    return jsonify(EXPERIMENTAL_RESULTS)


# ==========================================
# ORIGINAL MODEL API ROUTES (Enhanced with GPU)
# ==========================================

@app.route('/api/model-stats')
def model_stats():
    """Get model performance statistics for dashboard"""
    sys_info = get_system_info()
    
    return jsonify({
        'svm_accuracy': EXPERIMENTAL_RESULTS.get('test_accuracy', 0.99) * 100,
        'svm_f1_score': EXPERIMENTAL_RESULTS.get('test_f1', 0.95) * 100,
        'test_samples': EXPERIMENTAL_RESULTS.get('test_samples', 0),
        'train_samples': EXPERIMENTAL_RESULTS.get('train_samples', 0),
        'models_loaded': {
            'svm': SVM_MODEL is not None,
            'gpt2': GPT2_MODEL is not None
        },
        'gpu_info': {
            'available': sys_info.get('gpu_available', False),
            'name': sys_info.get('gpu_name', 'N/A'),
            'svm_on_gpu': sys_info.get('svm_on_gpu', False),
            'gpt2_on_gpu': sys_info.get('gpt2_on_gpu', False)
        }
    })

@app.route('/api/predict-difficulty', methods=['POST'])
def api_predict_difficulty():
    """Predict difficulty level of text (legacy endpoint)"""
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    result = predict_difficulty(text)
    return jsonify(result)

@app.route('/api/generate-content', methods=['POST'])
def api_generate_content():
    """Generate adaptive reading content with GPU acceleration"""
    data = request.json
    topic = data.get('topic', 'general knowledge')
    difficulty = data.get('difficulty', 'intermediate')
    
    # Generate passage (GPU-accelerated if available)
    passage = generate_passage(topic, difficulty)
    
    # Generate questions
    questions = generate_questions(passage)
    
    # Predict actual difficulty of generated content
    actual_difficulty = predict_difficulty(passage)
    
    return jsonify({
        'passage': passage,
        'questions': questions,
        'requested_difficulty': difficulty,
        'actual_difficulty': actual_difficulty,
        'topic': topic,
        'used_gpu': actual_difficulty.get('used_gpu', False)
    })

@app.route('/api/adaptive-quiz', methods=['POST'])
@login_required
def api_adaptive_quiz():
    """Generate adaptive quiz based on student performance"""
    data = request.json
    student_level = data.get('level', 3)  # 1-6 scale
    
    # Map level to difficulty
    difficulty_map = DIFFICULTY_MAP_LIST
    difficulty = difficulty_map.get(student_level, 'intermediate')
    
    # Generate content
    topics = ['science', 'history', 'technology', 'literature', 'environment']
    topic = np.random.choice(topics)
    
    passage = generate_passage(topic, difficulty)
    questions = generate_questions(passage)
    
    return jsonify({
        'passage': passage,
        'questions': questions,
        'difficulty': difficulty,
        'topic': topic,
        'student_level': student_level
    })

@app.route('/api/evaluate-answer', methods=['POST'])
@login_required
def api_evaluate_answer():
    """Evaluate student answer and update difficulty"""
    data = request.json
    answer = data.get('answer', '')
    correct_answer = data.get('correct_answer', '')
    current_level = data.get('current_level', 3)
    
    # Simple evaluation (can be enhanced with NLP)
    is_correct = answer.lower().strip() in correct_answer.lower()
    
    # Adaptive difficulty adjustment
    new_level = current_level
    if is_correct:
        new_level = min(6, current_level + 1)  # Increase difficulty
    else:
        new_level = max(1, current_level - 1)  # Decrease difficulty
    
    return jsonify({
        'is_correct': is_correct,
        'current_level': current_level,
        'new_level': new_level,
        'feedback': 'Correct! Moving to harder content.' if is_correct else 'Let\'s try something easier.'
    })



# ==========================================
# WEB ROUTES
# ==========================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    """Dashboard with GPU-aware model stats"""
    try:
        # Get model stats with GPU info
        stats_response = model_stats()
        stats = json.loads(stats_response.get_data(as_text=True))
        
        # Generate personalized content
        user_level = 3  # TODO: Get from user profile in database
        
        # Generate quiz content
        difficulty_map = DIFFICULTY_MAP_LIST
        difficulty = difficulty_map.get(user_level, 'intermediate')
        
        topics = ['science', 'history', 'technology', 'literature', 'environment']
        topic = np.random.choice(topics)
        
        passage = generate_passage(topic, difficulty)
        questions = generate_questions(passage)
        
        content = {
            'passage': passage,
            'questions': questions,
            'difficulty': difficulty,
            'topic': topic,
            'student_level': user_level
        }
        
        return render_template('dashboard.html', 
                             model_stats=stats,
                             quiz_content=content,
                             gpu_available=stats['gpu_info']['available'])
    except Exception as e:
        print(f"Error in dashboard: {e}")
        flash('Error loading dashboard content', 'error')
        return render_template('dashboard.html', 
                             model_stats={},
                             quiz_content={},
                             gpu_available=False)

@app.route('/analytics')
@login_required
def analytics():
    """Analytics page with experimental results"""
    try:
        # Load experimental results
        results_path = Path('results/experimental_results.json')
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)
        else:
            results = EXPERIMENTAL_RESULTS
        
        # Get GPU system info
        sys_info = get_system_info()
        
        return render_template('analytics.html', 
                             experimental_results=results,
                             system_info=sys_info)
    except Exception as e:
        print(f"Error in analytics: {e}")
        flash('Error loading analytics', 'error')
        return render_template('analytics.html', 
                             experimental_results={},
                             system_info={})

@app.route('/generate-quiz')
@login_required
def generate_quiz():
    """Generate adaptive quiz page with GPU-accelerated content"""
    return render_template('quiz.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            flash('Logged in successfully!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'error')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirmPassword')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('signup.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'error')
        else:
            hashed_password = generate_password_hash(password)
            new_user = User(name=name, email=email, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            
            flash('Account created successfully! Please login.', 'success')
            return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully', 'success')
    return redirect(url_for('index'))


# ==========================================
# ADMIN ROUTES
# ==========================================

@app.route('/setup-admin')
def setup_admin():
    try:
        # Check if admin user already exists
        admin = User.query.filter_by(email='admin@readapt.com').first()
        if not admin:
            hashed_password = generate_password_hash('admin123')
            admin_user = User(
                name='Administrator',
                email='admin@readapt.com',
                password=hashed_password,
                is_admin=True
            )
            db.session.add(admin_user)
            db.session.commit()
            flash('Admin user created successfully! Email: admin@readapt.com, Password: admin123', 'success')
            print("✅ Admin user created successfully!")
        else:
            flash('Admin user already exists!', 'warning')
            print("ℹ️ Admin user already exists")
        
        return redirect(url_for('login'))
    except Exception as e:
        flash(f'Error creating admin user: {str(e)}', 'error')
        print(f"❌ Error: {e}")
        return redirect(url_for('index'))

@app.route('/admin')
@login_required
def admin_dashboard():
    """Admin dashboard with GPU system monitoring"""
    # Check if user is admin
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('dashboard'))
    
    # Get system info including GPU
    sys_info = get_system_info()
    
    # Get model stats
    stats_response = model_stats()
    stats = json.loads(stats_response.get_data(as_text=True))
    
    # Sample data for the admin dashboard
    recent_users = [
        {'name': 'John Doe', 'email': 'john@example.com', 'progress': 75, 'status': 'active'},
        {'name': 'Jane Smith', 'email': 'jane@example.com', 'progress': 42, 'status': 'active'},
        {'name': 'Mike Johnson', 'email': 'mike@example.com', 'progress': 90, 'status': 'inactive'},
        {'name': 'Sarah Wilson', 'email': 'sarah@example.com', 'progress': 63, 'status': 'active'},
        {'name': 'Tom Brown', 'email': 'tom@example.com', 'progress': 28, 'status': 'active'}
    ]
    
    courses = [
        {'name': 'Introduction to Programming', 'icon': 'fas fa-code', 'enrolled': 342, 'completion': 85},
        {'name': 'Data Structures', 'icon': 'fas fa-project-diagram', 'enrolled': 287, 'completion': 72},
        {'name': 'Web Development', 'icon': 'fas fa-laptop-code', 'enrolled': 156, 'completion': 68},
        {'name': 'Database Design', 'icon': 'fas fa-database', 'enrolled': 198, 'completion': 79}
    ]
    
    recent_activity = [
        {'type': 'user', 'icon': 'user-plus', 'message': 'New user registration: Alex Thompson', 'time': '5 minutes ago'},
        {'type': 'course', 'icon': 'book', 'message': 'Course completed: Introduction to Programming', 'time': '12 minutes ago'},
        {'type': 'system', 'icon': 'cog', 'message': 'System backup completed successfully', 'time': '1 hour ago'},
        {'type': 'payment', 'icon': 'credit-card', 'message': 'Payment received from Sarah Wilson', 'time': '2 hours ago'},
        {'type': 'support', 'icon': 'life-ring', 'message': 'Support ticket #1245 resolved', 'time': '3 hours ago'}
    ]
    
    return render_template('admin_dashboard.html', 
                         recent_users=recent_users,
                         courses=courses,
                         recent_activity=recent_activity,
                         system_info=sys_info,
                         model_stats=stats)

@app.route('/dashboard-enhanced')
@login_required
def dashboard_enhanced():
    """Enhanced dashboard with pre-generated content"""
    try:
        # Get model stats with GPU info
        stats_response = model_stats()
        stats = json.loads(stats_response.get_data(as_text=True))
        
        # Generate personalized content sample
        user_level = 3  # TODO: Get from user profile in database
        
        difficulty_map = DIFFICULTY_MAP_LIST
        difficulty = difficulty_map.get(user_level, 'intermediate')
        
        topics = ['science', 'history', 'technology', 'literature', 'environment']
        topic = np.random.choice(topics)
        
        # Generate a preview passage
        preview_passage = generate_passage(topic, difficulty, max_length=150)
        
        content = {
            'preview_passage': preview_passage,
            'preview_topic': topic,
            'difficulty': difficulty,
            'student_level': user_level
        }
        
        return render_template('dashboard.html', 
                             model_stats=stats,
                             content=content,
                             gpu_available=stats['gpu_info']['available'])
    except Exception as e:
        print(f"Error in enhanced dashboard: {e}")
        flash('Error loading dashboard content', 'error')
        return render_template('dashboard.html', 
                             model_stats={},
                             content={},
                             gpu_available=False)


# ==========================================
# INITIALIZATION
# ==========================================

def init_db():
    """Initialize database tables"""
    with app.app_context():
        db.create_all()
        print("✅ Database tables created!")

def run_data_pipeline():
    """Run data pipeline if needed"""
    processed_empty = is_folder_empty_pathlib('processed_data')
    datasets_empty = is_folder_empty_pathlib('datasets')
    
    if processed_empty or datasets_empty:
        print("\n🔄 Running data pipeline...")
        os.system("python data_pipeline.py")
    else:
        print("✅ Data already processed, skipping pipeline")

def print_startup_info():
    """Print startup information including GPU status"""
    print("\n" + "=" * 60)
    print("READAPT ADAPTIVE LEARNING SYSTEM")
    print("=" * 60)
    
    if GPU_AVAILABLE:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎮 GPU: {gpu_name}")
        print(f"💾 GPU Memory: {gpu_memory:.2f} GB")
    else:
        print("💻 Running on CPU")
    
    print(f"✅ SVM Model: {'Loaded' if SVM_MODEL else 'Not Loaded'}")
    print(f"✅ GPT-2 Model: {'Loaded' if GPT2_MODEL else 'Not Loaded'}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    init_db()
    print_startup_info()
    request.url('/setup-admin')
    app.run(debug=True)
    # run_data_pipeline()  # Uncomment if you want to run on startup
