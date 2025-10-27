from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'readapt-secret-key-2024'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///readapt.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# User Model
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
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/analytics')
@login_required
def analytics():
    return render_template('analytics.html')

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
            hashed_password = generate_password_hash(password, method='sha256')
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

# Admin Routes
@app.route('/setup-admin')
def setup_admin():
    try:
        # Check if admin user already exists
        admin = User.query.filter_by(email='admin@readapt.com').first()
        if not admin:
            hashed_password = generate_password_hash('admin123', method='sha256')
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
    # Check if user is admin
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('dashboard'))
    
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
                         recent_activity=recent_activity)

# Initialize database
def init_db():
    with app.app_context():
        db.create_all()
        print("✅ Database tables created!")

if __name__ == '__main__':
    init_db()
    app.run(debug=True)