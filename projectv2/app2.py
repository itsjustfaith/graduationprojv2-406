import os
# Completely suppress FFmpeg logging
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '8'  # Only fatal errors
os.environ['GST_DEBUG'] = '0'  # Suppress GStreamer logs

from flask import Flask, render_template, Response, redirect, url_for, request, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from flask_migrate import Migrate
from ultralytics import YOLO  # Import YOLOv8
import base64
import re
import torch 
import torch.backends.cudnn as cudnn  # For CUDA optimization
import warnings
from contextlib import redirect_stderr, nullcontext
import io


# DETECTION LIBRARIES
from flask import Flask, render_template, request, redirect, url_for, Response
from flask_socketio import SocketIO, emit
import cv2
import os
import numpy as np

# Initialize Flask app and configurations
app = Flask(__name__)
socketio = SocketIO(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'  # SQLite database file
app.config['SECRET_KEY'] = 'mysecretkey'  # Secret key for Flask session management
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disable modification tracking

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Initialize Flask-Migrate
migrate = Migrate(app, db)

# Initialize Flask-Login
login_manager = LoginManager(app)
login_manager.login_view = 'login'  # Set the login view for Flask-Login

# Define the User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)  # Add admin flag

    def __repr__(self):
        return f'<User {self.email}>'

# Define DetectionLog model
class DetectionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(20), nullable=False)
    time = db.Column(db.String(20), nullable=False)
    used_resource = db.Column(db.String(50), nullable=False)
    total_count = db.Column(db.Integer, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref='detection_logs')

    def __repr__(self):
        return f'<DetectionLog {self.date} {self.time} {self.used_resource} {self.total_count}>'

def reset_auto_increment():
    db.session.execute('DELETE FROM sqlite_sequence WHERE name="detection_log";')
    db.session.execute('DELETE FROM sqlite_sequence WHERE name="user";')
    db.session.commit()

# Load user function for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))  # Updated to use session.get()

# Admin login check
def is_admin(user):
    return user.is_authenticated and user.is_admin

#Landing Page route
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login_signup.html')

@app.route('/login_signup', methods=['GET', 'POST'])
def login_signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form.get('confirm_password')
        username = request.form.get('username')

        user = User.query.filter_by(email=email).first()

        if confirm_password:  # Signup logic
            if user:
                flash('Email already exists! Please log in.', 'signup_error')
                return redirect(url_for('login_signup', form='signup'))
            elif password != confirm_password:
                flash('Passwords do not match!', 'signup_error')
                return redirect(url_for('login_signup', form='signup'))
            else:
                is_admin = False
                if User.query.count() == 0:  # First user becomes admin
                    is_admin = True
                
                hashed_password = generate_password_hash(password)
                new_user = User(
                    email=email,
                    password=hashed_password,
                    username=username,
                    is_admin=is_admin
                )
                db.session.add(new_user)
                db.session.commit()
                
                if is_admin:
                    login_user(new_user)
                    session['admin_logged_in'] = True
                    session['admin_username'] = new_user.username
                    flash('Admin account created successfully!', 'signup_success')
                    return redirect(url_for('admin'))  # Redirect to admin dashboard
                
                flash('Account created successfully! Please Sign in.', 'signup_success')
                return redirect(url_for('login_signup', form='signup'))
                
        else:  # Login logic
            if user and check_password_hash(user.password, password):
                login_user(user)
                if user.is_admin:
                    session['admin_logged_in'] = True
                    session['admin_username'] = user.username
                    return redirect(url_for('admin'))  # Explicit admin redirect
                return redirect(url_for('user_dash'))
            else:
                flash('Wrong Email or Password!', 'login_error')
                return redirect(url_for('login_signup', form='signin'))

    return render_template('login_signup.html')
    
# @app.route('/admin')
# # @login_required  # Comment out this line
# def admin():
#     # Auto-login logic
#     if not current_user.is_authenticated:
#         admin = User.query.filter_by(is_admin=True).first()
#         if admin:
#             login_user(admin)
    
#     return render_template('admin.html')

#Admin routes
@app.route('/admin')
@login_required
def admin():
    if not current_user.is_admin:
        flash('You do not have admin privileges', 'error')
        return redirect(url_for('user_dash'))
    return render_template('admin.html')

@app.route('/admin_detection')
@login_required
def admin_detection():
    if not current_user.is_admin:
        flash('You do not have admin privileges', 'error')
        return redirect(url_for('user_dash'))
    return render_template('admin_detection.html')

# API endpoints for admin pages
@app.route('/api/users', methods=['GET', 'POST'])
def api_users():
    if request.method == 'GET':
        users = User.query.all()
        return jsonify([{
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'is_admin': user.is_admin
        } for user in users])
    elif request.method == 'POST':
        data = request.get_json()
        # Validate required fields
        if not data or not data.get('username') or not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Missing required fields'}), 400
            
        # Check if username already exists
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'error': 'Username already exists'}), 400
            
        # Check if email already exists
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'Email already exists'}), 400
            
        # Create new user
        hashed_password = generate_password_hash(data['password'])
        new_user = User(
            username=data['username'],
            email=data['email'],
            password=hashed_password,
            is_admin=data.get('is_admin', False)
        )
        db.session.add(new_user)
        db.session.commit()
        
        return jsonify({
            'message': 'User created successfully',
            'user': {
                'id': new_user.id,
                'username': new_user.username,
                'email': new_user.email,
                'is_admin': new_user.is_admin
            }
        }), 201
    
@app.route('/api/users/<int:user_id>', methods=['PUT', 'DELETE'])
def api_user(user_id):
    user = User.query.get_or_404(user_id)
    if request.method == 'PUT':
        data = request.get_json()
        
        # Check if username is being changed and validate
        if 'username' in data and data['username'] != user.username:
            if User.query.filter(User.username == data['username']).first():
                return jsonify({'error': 'Username already exists'}), 400
            user.username = data['username']
        
        # Check if email is being changed and validate
        if 'email' in data and data['email'] != user.email:
            if User.query.filter(User.email == data['email']).first():
                return jsonify({'error': 'Email already exists'}), 400
            user.email = data['email']
        
        # Update password if provided
        if 'password' in data and data['password']:
            user.password = generate_password_hash(data['password'])
        
        # Update admin status
        if 'is_admin' in data:
            user.is_admin = data['is_admin']
            
        db.session.commit()
        return jsonify({
            'message': 'User updated successfully',
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_admin': user.is_admin
            }
        })
    elif request.method == 'DELETE':
        db.session.delete(user)
        db.session.commit()
        return jsonify({'message': 'User deleted successfully'})

@app.route('/api/users/delete-multiple', methods=['POST'])
def api_delete_multiple_users():
    data = request.get_json()
    if not data or not data.get('userIds'):
        return jsonify({'error': 'No user IDs provided'}), 400
        
    try:
        User.query.filter(User.id.in_(data['userIds'])).delete(synchronize_session=False)
        db.session.commit()
        return jsonify({'message': f"{len(data['userIds'])} users deleted successfully"})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/detection-logs', methods=['GET'])
def api_detection_logs():
    logs = DetectionLog.query.all()
    return jsonify([{
        'id': log.id,
        'user_id': log.user_id,
        'date': log.date,
        'time': log.time,
        'total_count': log.total_count,
        'used_resource': log.used_resource
    } for log in logs])

@app.route('/api/detection-logs/<int:log_id>', methods=['DELETE'])
def api_detection_log(log_id):
    log = DetectionLog.query.get_or_404(log_id)
    db.session.delete(log)
    db.session.commit()
    return jsonify({'message': 'Log deleted successfully'})

@app.route('/api/detection-logs/delete-multiple', methods=['POST'])
def api_delete_multiple_logs():
    data = request.get_json()
    if not data or not data.get('logIds'):
        return jsonify({'error': 'No log IDs provided'}), 400
        
    try:
        DetectionLog.query.filter(DetectionLog.id.in_(data['logIds'])).delete(synchronize_session=False)
        db.session.commit()
        return jsonify({'message': f"{len(data['logIds'])} logs deleted successfully"})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
    
# Logout route
@app.route('/logout')
def logout():
    if current_user.is_authenticated:
        logout_user()
    session.clear()  # Clear all session data
    return redirect(url_for('login_signup'))

# Detection configuration
UPLOAD_FOLDER = 'static/uploads'
AUDIO_FOLDER = 'static/audio' #For Audio
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['AUDIO_FOLDER'] = AUDIO_FOLDER #For Audio

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(AUDIO_FOLDER):  #For Audio
    os.makedirs(AUDIO_FOLDER)

# Global variables for people count and control signals
people_count = 0
max_people_count = 0
stop_feed = False

# Initialize YOLO model
def initialize_yolo():
    try:
        # Initialize torch settings
        if torch.cuda.is_available():
            cudnn.enabled = True
            cudnn.benchmark = True
            torch.set_flush_denormal(True)
        
        # Load model with error handling
        model_path = "yolov8n.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        model = YOLO(model_path)
        model.fuse()  # Optimize model
        return model
        
    except Exception as e:
        print(f"YOLO initialization failed: {str(e)}")
        return None
    
heatmap_data = []

# I Modified detect_people function to store coordinates
def detect_people(frame, model):
    global people_count, heatmap_data
    try:
        if frame is None or model is None:
            print("Invalid frame or model")
            return [], []
            
        results = model(frame, stream=True, verbose=False)
        bounding_boxes = []
        confidences = []
        current_frame_data = []
        
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                cls_id = box.cls.item() if hasattr(box.cls, 'item') else int(box.cls)
                conf = box.conf.item() if hasattr(box.conf, 'item') else float(box.conf)
                if cls_id == 0 and conf > 0.4:  # Class 0 is person
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    bounding_boxes.append([x1, y1, x2-x1, y2-y1])
                    confidences.append(conf)
                    
                    # Store center point of each detection with integer values
                    center_x = int((x1 + x2) // 2)
                    center_y = int((y1 + y2) // 2)
                    current_frame_data.append((center_x, center_y))
        
        people_count = len(bounding_boxes)
        
        # Store this frame's data only if people were detected
        if current_frame_data:
            heatmap_data.append(current_frame_data)
            # Keep only the last 10 frames of heatmap data
            heatmap_data = heatmap_data[-10:]  
            
        return bounding_boxes, confidences
    
    except Exception as e:
        print(f"Detection error: {str(e)}")
        return [], []

@app.route('/heatmap_data')
def get_heatmap_data():
    global heatmap_data
    try:
        all_points = []
        for frame in heatmap_data:
            for x, y in frame:
                all_points.append({
                    'x': int(x),  
                    'y': int(y),
                    'frame_width': 640,
                    'frame_height': 480
                })
        return jsonify(all_points)
    except Exception as e:
        app.logger.error(f"Error in heatmap_data: {str(e)}")
        return jsonify([]), 200  # Return empty array instead of error

# Generate video frames with detection
def generate_frames(video_source=None):
    global stop_feed, max_people_count
    
    # Use a more robust temporary file handling
    import tempfile
    import sys
    from contextlib import contextmanager
    
    @contextmanager
    def suppress_stderr():
        original_stderr = sys.stderr
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                sys.stderr = f
                yield
        finally:
            sys.stderr = original_stderr
            try:
                if os.path.exists(f.name):
                    os.unlink(f.name)  # More reliable than remove on Windows
            except:
                pass  # Ignore any cleanup errors

    with suppress_stderr():
        model = initialize_yolo()
        if model is None:
            print("Failed to initialize YOLO model")
            return
        
        try:
            # If no specific source was set, try default sources
            if video_source is None:
                video_sources = [1, 0]  # Default camera indices
                cap = None
                for source in video_sources:
                    cap = cv2.VideoCapture(source)
                    if cap.isOpened():
                        print(f"Using video source: {source}")
                        break
                    cap.release()
            else:
                # Use the provided video source (IP camera)
                cap = cv2.VideoCapture(video_source)
                print(f"Using IP camera: {video_source}")

            if not cap or not cap.isOpened():
                print("No valid camera source found.")
                return

            try:
                while not stop_feed:
                    success, frame = cap.read()
                    if not success:
                        break
                    
                    boxes, confidences = detect_people(frame, model)
                    max_people_count = max(max_people_count, people_count)
                   
                    for i, (x, y, w, h) in enumerate(boxes):
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, f"Person {confidences[i]:.2f}", 
                                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    ret, buffer = cv2.imencode('.jpg', frame)
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    
            except GeneratorExit:
                # Handle graceful shutdown when client disconnects
                pass
                
            finally:
                cap.release()
                
        except Exception as e:
            print(f"Video processing error: {str(e)}")

# Handle image upload
@app.route('/upload_image', methods=['POST'])
@login_required
def upload_image():
    global stop_feed
    stop_feed = True
    image = request.files['image']
    if image:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(image_path)
        return redirect(url_for('display_image', image_path=image_path))
    return redirect(url_for('user_dash'))

# Display image with detection
@app.route('/display_image')
@login_required
def display_image():
    image_path = request.args.get('image_path')
    frame = cv2.imread(image_path)
    model = initialize_yolo()
    boxes, confidences = detect_people(frame, model)
    
    for i, box in enumerate(boxes):
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Person {confidences[i]:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imwrite(image_path, frame)

    if current_user.is_authenticated:
        log = DetectionLog(
            date=datetime.now().strftime("%Y-%m-%d"),
            time=datetime.now().strftime("%H:%M:%S"),
            used_resource='Image-Upload',
            total_count=people_count,
            user_id=current_user.id
        )
        db.session.add(log)
        db.session.commit()

    image_paths = image_path.split('/')
    for p in image_paths:
        if "\\" in p:
            image_paths.remove(p)
            image_paths = image_paths + p.split('\\')
    
    return render_template('detect.html', 
        image_path='/'.join(image_paths), 
        people_count=people_count,
        is_static=True
    )

# Handle video upload
@app.route('/upload_video', methods=['POST'])
@login_required
def upload_video():
    global stop_feed
    stop_feed = True
    video = request.files['video']
    if video:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
        video.save(video_path)
        return redirect(url_for('display_video', video_filename=video.filename))
    return redirect(url_for('user_dash'))

# Display video with detection
@app.route('/display_video')
@login_required
def display_video():
    video_filename = request.args.get('video_filename')
    processed_video_filename = 'processed_' + video_filename
    input_video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        return "Error: Could not open video file"
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_video_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_video_filename)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    model = initialize_yolo()
    max_count = 0
    frame_count = 0
    last_boxes = []
    last_confidences = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 3 == 0:
            boxes, confidences = detect_people(frame, model)
            max_count = max(max_count, len(boxes))
            last_boxes = boxes
            last_confidences = confidences
        
        for i, box in enumerate(last_boxes):
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {last_confidences[i]:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        out.write(frame)
    
    cap.release()
    out.release()

    if current_user.is_authenticated:
        log = DetectionLog(
            date=datetime.now().strftime("%Y-%m-%d"),
            time=datetime.now().strftime("%H:%M:%S"),
            used_resource='Video-Upload',
            total_count=max_count,
            user_id=current_user.id
        )
        db.session.add(log)
        db.session.commit()

    processed_video_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_video_filename)
    processed_paths = processed_video_path.split('/')
    for p in processed_paths:
        if "\\" in p:
            processed_paths.remove(p)
            processed_paths = processed_paths + p.split('\\')
    
    return render_template('detect.html', 
        video_path='/'.join(processed_paths),
        people_count=max_count,
        is_static=True
    )

# User Dashboard route
@app.route('/user_dash')
@login_required
def user_dash():
    global stop_feed
    stop_feed = False
    resource_type = request.form.get('resource_type')
    session['used_resource'] = resource_type
    return render_template('user_dash.html')

@app.route('/user_profile', methods=['GET', 'POST'])
@login_required
def user_profile():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        user = User.query.get(current_user.id)
        
        if username and username != user.username:
            existing_user = User.query.filter_by(username=username).first()
            if existing_user and existing_user.id != current_user.id:
                flash('Username already exists!', 'error')
            else:
                user.username = username
                flash('Username updated successfully!', 'success')
        
        if email and email != user.email:
            existing_user = User.query.filter_by(email=email).first()
            if existing_user and existing_user.id != current_user.id:
                flash('Email already exists!', 'error')
            else:
                user.email = email
                flash('Email updated successfully!', 'success')
        
        if current_password and new_password and confirm_password:
            if check_password_hash(user.password, current_password):
                if new_password == confirm_password:
                    user.password = generate_password_hash(new_password)
                    flash('Password updated successfully!', 'success')
                else:
                    flash('New passwords do not match!', 'error')
            else:
                flash('Current password is incorrect!', 'error')
        
        db.session.commit()
        return redirect(url_for('user_profile'))
    
    return render_template('user_profile.html', user=current_user)

# Detection route
@app.route('/detect', methods=['GET', 'POST'])
@login_required
def detect():
    global stop_feed, people_count, max_people_count
    people_count = 0
    max_people_count = 0
    stop_feed = False

    if request.method == 'POST':
        resource_type = request.form.get('resource_type')
        session['used_resource'] = resource_type

        if resource_type == 'ip_camera':
            ip_address = request.form.get('ip_address')
            # Formatting the IP address into a proper RTSP URL
            rtsp_url = f"rtsp://{ip_address}:554/stream1"  # Adjust this based on your camera's RTSP URL format
            stop_feed = False
            socketio.start_background_task(background_people_count)
            return render_template('detect.html', resource_type='live', people_count=people_count, video_source=rtsp_url)

    resource_type = request.form.get('resource_type')
    if resource_type == 'live':
        stop_feed = False
        socketio.start_background_task(background_people_count)
        return render_template('detect.html', resource_type='live', people_count=people_count)
    elif resource_type == 'image':
        return redirect(url_for('upload_image'))
    elif resource_type == 'video':
        return redirect(url_for('upload_video'))

    people_count = 0
    stop_feed = False
    socketio.start_background_task(background_people_count)
    return render_template('detect.html', is_static=False)

@app.route('/check_audio')
def check_audio():
    audio_path = os.path.join(app.config['AUDIO_FOLDER'], 'alert.mp3')
    if os.path.exists(audio_path):
        return f"Audio file exists at {audio_path}"
    else:
        return f"Audio file not found at {audio_path}"

# Video feed route
@app.route('/video_feed')
def video_feed():
    video_source = request.args.get('source')
    if video_source == 'None':
        video_source = None
    return Response(generate_frames(video_source), mimetype='multipart/x-mixed-replace; boundary=frame')

# Stop feed route
@app.route('/stop_feed', methods=['POST'])
@login_required
def stop_feed_func():
    global stop_feed, people_count, max_people_count, heatmap_data
    stop_feed = True
    heatmap_data = []  # Clear heatmap data when feed stops
    socketio.emit('feed_stopped')

    import time
    time.sleep(0.5)  # Allow 500ms for cleanup

    used_resource = session.get('used_resource', 'Live Feed')
    if used_resource == 'live':
        used_resource = 'Live-Feed'

    if current_user.is_authenticated:
        log = DetectionLog(
            date=datetime.now().strftime("%Y-%m-%d"),
            time=datetime.now().strftime("%H:%M:%S"),
            used_resource=used_resource,
            total_count=max_people_count,
            user_id=current_user.id
        )
        db.session.add(log)
        db.session.commit()
    
    people_count = 0
    max_people_count = 0
    return redirect(url_for('user_dash'))

# Background task for people count
def background_people_count():
    global stop_feed
    while not stop_feed:
        socketio.emit('people_count', {'count': people_count })
        socketio.sleep(1)

# User count route
@app.route('/user_count')
@login_required
def user_count():
    logs = DetectionLog.query.filter_by(user_id=current_user.id).all()
    return render_template('user_count.html', logs=logs)

# Delete log route
@app.route('/delete_log/<int:log_id>')
@login_required
def delete_log(log_id):
    db.session.execute('PRAGMA foreign_keys = OFF;')
    try:
        log = DetectionLog.query.get(log_id)
        if log:
            db.session.delete(log)
            db.session.commit()
    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting record: {e}", 'error')
    finally:
        db.session.execute('PRAGMA foreign_keys = ON;')
        db.session.commit()
        reset_auto_increment()
    return redirect(url_for('user_count'))

def ensure_admin_exists():
    with app.app_context():
        if User.query.count() == 0:  # If no users exist
            admin = User(
                username='admin',
                email='admin@example.com',
                password=generate_password_hash('admin123'),
                is_admin=True
            )
            db.session.add(admin)
            db.session.commit()

# Initialize database
if __name__ == '__main__':
    # Suppress all warnings
    import warnings
    warnings.filterwarnings("ignore")

    with app.app_context():
        db.create_all()
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)