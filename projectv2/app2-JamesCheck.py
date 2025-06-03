# Standard Library
import os
import re
import base64
import warnings
from datetime import datetime
from contextlib import redirect_stderr, nullcontext
import io

# Import Flask modules 
from flask import Flask, render_template, Response, redirect, url_for, request, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_migrate import Migrate
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from ultralytics import YOLO

#Frontend - This suppresses logs in terminal for testing purposes
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '8'  
os.environ['GST_DEBUG'] = '0'  


#Frontend - init flask and socketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Note: we are using SQLite
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
app.config['SECRET_KEY'] = 'mysecretkey'  
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False 

# DB code 
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# User login logic 
login_manager = LoginManager(app)
login_manager.login_view = 'login'  # Redirects to login


# --- DATABASE THINGS ---
# We are using SQLAlchemy ORM for our users
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False) 
    email = db.Column(db.String(120), unique=True, nullable=False)  
    password = db.Column(db.String(200), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)  

    def __repr__(self):
        return f"<User email='{self.email}'>"  # for debugging

# Track all the detection going on
class DetectionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(20), nullable=False)  
    time = db.Column(db.String(20), nullable=False)
    used_resource = db.Column(db.String(50), nullable=False)  # to know whether it is a camera, image, video for our system
    total_count = db.Column(db.Integer, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref='logs_that_this_user_owns')

    def __repr__(self):
        return f"<DetectionLog when={self.date} @ {self.time} | method={self.used_resource} | count={self.total_count}>"

def reset_auto_increment():
    try:
        db.session.execute('DELETE FROM sqlite_sequence WHERE name="detection_log";')
        db.session.execute('DELETE FROM sqlite_sequence WHERE name="user";')
        db.session.commit()
    except Exception as e:
        print(f"could not reset auto-increments: {e}")  


# to help flask figure out user in dencos session
@login_manager.user_loader
def load_user(uid):
    try:
        return db.session.get(User, int(uid))  
    except:
        return None  

def is_admin(user_to_check):
    return user_to_check.is_authenticated and user_to_check.is_admin  

# --- LANDING & AUTH ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login')
def login():
    # frontend team: We use one template for both login and signup
    return render_template('login_signup.html')


@app.route('/login_signup', methods=['GET', 'POST'])
def login_signup():
    if request.method == 'POST':
        
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')  # only present during signup
        username = request.form.get('username') 

        user = User.query.filter_by(email=email).first()

        if confirm_password:  # a signup attempt
            if user:
                flash("Email already in use", 'signup_error')
                return redirect(url_for('login_signup', form='signup'))

            if password != confirm_password:
                flash("Passwords don't match. Please try again.", 'signup_error')
                return redirect(url_for('login_signup', form='signup'))

            # Automatically grant admin status to first user in denco db
            first_user = User.query.count() == 0
            make_admin = True if first_user else False

            hashed_pw = generate_password_hash(password)  
            new_user = User(
                email=email,
                password=hashed_pw,
                username=username or "anonymous_ghost",  
                is_admin=make_admin
            )

            db.session.add(new_user)
            db.session.commit()

            if make_admin:
                login_user(new_user)
                session['admin_logged_in'] = True
                session['admin_username'] = new_user.username
                flash(" Admin account created!", 'signup_success')
                return redirect(url_for('admin'))

            flash(" Your account is ready. Please sign in.", 'signup_success')
            return redirect(url_for('login_signup', form='signup'))

        else: 
            if user and check_password_hash(user.password, password):
                login_user(user)
                if user.is_admin:
                    session['admin_logged_in'] = True
                    session['admin_username'] = user.username
                    return redirect(url_for('admin'))  

                return redirect(url_for('user_dash'))
            else:
                flash(" Wrong email or password. Try again?", 'login_error')
                return redirect(url_for('login_signup', form='signin'))

    return render_template('login_signup.html')

# --- ADMIN AREA ---

@app.route('/admin')
@login_required
def admin():
    if not current_user.is_admin:
        flash("Not an admin!", 'error')
        return redirect(url_for('user_dash'))

    return render_template('admin.html')


@app.route('/admin_detection')
@login_required
def admin_detection():
    if not current_user.is_admin:
        flash("To be accessed by admin", 'error')
        return redirect(url_for('user_dash'))

    return render_template('admin_detection.html')


# --- USER API  ---

@app.route('/api/users', methods=['GET', 'POST'])
def api_users():
    if request.method == 'GET':
        # Return ALL the users — 
        all_users = User.query.all()
        return jsonify([
            {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_admin': user.is_admin
            }
            for user in all_users
        ])

    elif request.method == 'POST':
        data = request.get_json()
        if not data or not data.get('username') or not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Missing fields... fill all the fields, please.'}), 400

        # Clash detection
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'error': 'That username is already taken.'}), 400

        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'Email is already in use. Try logging in?' }), 400

        #  make new user
        hashed = generate_password_hash(data['password'])
        created_user = User(
            username=data['username'],
            email=data['email'],
            password=hashed,
            is_admin=data.get('is_admin', False)
        )
        db.session.add(created_user)
        db.session.commit()

        return jsonify({
            'message': 'New user created',
            'user': {
                'id': created_user.id,
                'username': created_user.username,
                'email': created_user.email,
                'is_admin': created_user.is_admin
            }
        }), 201


@app.route('/api/users/<int:user_id>', methods=['PUT', 'DELETE'])
def api_user(user_id):
    user = User.query.get_or_404(user_id)

    if request.method == 'PUT':
        updates = request.get_json()

        # modified and works now 
        if 'username' in updates and updates['username'] != user.username:
            if User.query.filter(User.username == updates['username']).first():
                return jsonify({'error': 'Username already taken'}), 400
            user.username = updates['username']

        if 'email' in updates and updates['email'] != user.email:
            if User.query.filter(User.email == updates['email']).first():
                return jsonify({'error': 'Email conflict'}), 400
            user.email = updates['email']

        if 'password' in updates and updates['password']:
            user.password = generate_password_hash(updates['password'])

        if 'is_admin' in updates:
            user.is_admin = updates['is_admin']

        db.session.commit()
        return jsonify({
            'message': 'User information updated',
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
        return jsonify({'message': f'User {user_id} deleted'})

# --- Crud operations DELETE ZONE(api points)  ---

@app.route('/api/users/delete-multiple', methods=['POST'])
def api_delete_multiple_users():
    payload = request.get_json()

    if not payload or not payload.get('userIds'):
        return jsonify({'error': 'Please provide at least *one* user ID...'}), 400

    try:
        # Execute user deletion in batches
        User.query.filter(User.id.in_(payload['userIds'])).delete(synchronize_session=False)
        db.session.commit()
        return jsonify({'message': f"{len(payload['userIds'])} users cleared."})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f"Try again: {str(e)}"}), 500


# --- DETECTION LOG STUFF(api points) ---

@app.route('/api/detection-logs', methods=['GET'])
def api_detection_logs():
    logs = DetectionLog.query.all()

    return jsonify([
        {
            'id': entry.id,
            'user_id': entry.user_id,
            'date': entry.date,
            'time': entry.time,
            'total_count': entry.total_count,
            'used_resource': entry.used_resource
        }
        for entry in logs
    ])


@app.route('/api/detection-logs/<int:log_id>', methods=['DELETE'])
def api_detection_log(log_id):
    log = DetectionLog.query.get_or_404(log_id)
    db.session.delete(log)
    db.session.commit()
    return jsonify({'message': f'Detection log {log_id} deleted. '})


@app.route('/api/detection-logs/delete-multiple', methods=['POST'])
def api_delete_multiple_logs():
    data = request.get_json()

    if not data or not data.get('logIds'):
        return jsonify({'error': 'List log IDs to delete...'}), 400

    try:
        DetectionLog.query.filter(DetectionLog.id.in_(data['logIds'])).delete(synchronize_session=False)
        db.session.commit()
        return jsonify({'message': f"{len(data['logIds'])} logs wiped off detection log."})
    except Exception as err:
        db.session.rollback()
        return jsonify({'error': f"Something went wrong: {str(err)}"}), 500

# --- end of SESSION ---

@app.route('/logout')
def logout():
    if current_user.is_authenticated:
        logout_user()  
    session.clear()  # clearing session data, Wipe it all, just to be sure
    return redirect(url_for('login_signup'))


# --- SOME STATIC FILE SETUP  ---

UPLOAD_FOLDER = 'static/uploads'
AUDIO_FOLDER = 'static/audio'  
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['AUDIO_FOLDER'] = AUDIO_FOLDER

# Check if folders exist; if not, Create required directories if not present
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)


# --- GLOBAL VARS ---
people_count = 0
max_people_count = 0
stop_feed = False
heatmap_data = []  #Temporary storage for tracking detection(heatmap) points on our system


# --- Setting up YOLO ---

def initialize_yolo():
    try:
        if torch.cuda.is_available():
            cudnn.enabled = True
            cudnn.benchmark = True
            torch.set_flush_denormal(True)

        model_path = "yolov8n.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f" Model missing at {model_path} — Raise error if model file is missing")

        model = YOLO(model_path)
        model.fuse()  
        return model
    except Exception as err:
        print(f"[YOLO failed to launch] {str(err)}")
        return None


# --- ACTUAL DETECTION ENGINE ---

def detect_people(frame, model):
    global people_count, heatmap_data

    if frame is None or model is None:
        print("Skip detection if input is invalid")
        return [], []

    try:
        results = model(frame, stream=True, verbose=False)
        bounding_boxes = []
        confidences = []
        temp_heatpoints = []

        for result in results:
            boxes = result.boxes.cpu().numpy()

            for box in boxes:
                class_id = int(box.cls) if not hasattr(box.cls, 'item') else box.cls.item()
                confidence = float(box.conf) if not hasattr(box.conf, 'item') else box.conf.item()

                if class_id == 0 and confidence > 0.4:
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    bounding_boxes.append([x1, y1, x2 - x1, y2 - y1])
                    confidences.append(confidence)

                    mid_x = int((x1 + x2) / 2)
                    mid_y = int((y1 + y2) / 2)
                    temp_heatpoints.append((mid_x, mid_y))

        people_count = len(bounding_boxes)

        if temp_heatpoints:
            heatmap_data.append(temp_heatpoints)
            # Retain only recent heatmap frames (last 10)
            heatmap_data = heatmap_data[-10:]

        return bounding_boxes, confidences

    except Exception as e:
        print(f"[ERROR in detect_people]: {e}")
        return [], []


# --- HEATMAP DATA ---

@app.route('/heatmap_data')
def get_heatmap_data():
    try:
        points = []
        for frame in heatmap_data:
            for (x, y) in frame:
                points.append({
                    'x': x,
                    'y': y,
                    'frame_width': 640,   # Static frame resolution values
                    'frame_height': 480
                })
        return jsonify(points)
    except Exception as err:
        app.logger.error(f"Heatmap error: {str(err)}")
        return jsonify([]), 200  # Send empty response if heatmap data is unavailable

# --- LIVE VIDEO(cam and IP)---

def generate_frames(video_source=None):
    global stop_feed, max_people_count

    import tempfile
    import sys
    from contextlib import contextmanager

    @contextmanager
    def suppress_stderr():
        # Suppress unnecessary OpenCV warnings
        original_stderr = sys.stderr
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                sys.stderr = f
                yield
        finally:
            sys.stderr = original_stderr
            try:
                if os.path.exists(f.name):
                    os.unlink(f.name)
            except:
                pass  

    with suppress_stderr():
        model = initialize_yolo()
        if model is None:
            print("No YOLO? No show.")
            return

        try:
            if video_source is None:
                cams_to_try = [1, 0]  # take note in that order - 1 for external cam not IP!
                cap = None
                for cam in cams_to_try:
                    cap = cv2.VideoCapture(cam)
                    if cap.isOpened():
                        print(f"Found a working webcam on source {cam}")
                        break
                    cap.release()
            else:
                print(f"Trying IP feed: {video_source}")
                cap = cv2.VideoCapture(video_source)

            if not cap or not cap.isOpened():
                print("Cannot access video.")
                return

            while not stop_feed:
                success, frame = cap.read()
                if not success:
                    break

                boxes, confidences = detect_people(frame, model)
                max_people_count = max(max_people_count, people_count)

                for i, (x, y, w, h) in enumerate(boxes):
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = f"Person {confidences[i]:.2f}"  
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (0, 255, 0), 2)

                # frontend : turn it into a JPEG before resulting it to the browser
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       buffer.tobytes() + b'\r\n')

        except GeneratorExit:
            pass

        except Exception as crash:
            print(f" Something went wrong during video processing: {crash}")

        finally:
            if cap:
                cap.release()

# --- HANDLE IMAGE UPLOADS---

@app.route('/upload_image', methods=['POST'])
@login_required
def upload_image():
    global stop_feed
    stop_feed = True  

    uploaded_img = request.files.get('image')

    if uploaded_img:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_img.filename)
        uploaded_img.save(file_path)
        return redirect(url_for('display_image', image_path=file_path))

    flash("No image uploaded!", "error")
    return redirect(url_for('user_dash'))


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
        label = f"Person {confidences[i]:.2f}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save the altered image to same path 
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

    parts = image_path.split('/')
    for p in parts:
        if "\\" in p:
            parts.remove(p)
            parts += p.split('\\')

    return render_template('detect.html',
        image_path='/'.join(parts),
        people_count=people_count,
        is_static=True)


# --- HANDLE VIDEO UPLOADS ---

@app.route('/upload_video', methods=['POST'])
@login_required
def upload_video():
    global stop_feed
    stop_feed = True

    vid = request.files.get('video')

    if vid:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], vid.filename)
        vid.save(file_path)
        return redirect(url_for('display_video', video_filename=vid.filename))

    flash("Video not found in form data. Please check again.", "error")
    return redirect(url_for('user_dash'))


@app.route('/display_video')
@login_required
def display_video():
    filename = request.args.get('video_filename')
    processed_name = 'processed_' + filename
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        return "Error: Could not open the selected video file"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_name)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    model = initialize_yolo()
    max_count = 0
    frame_number = 0
    last_boxes = []
    last_confidences = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1

        if frame_number % 3 == 0:
            last_boxes, last_confidences = detect_people(frame, model)
            max_count = max(max_count, len(last_boxes))

        for i, box in enumerate(last_boxes):
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {last_confidences[i]:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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

    # Normalize paths 
    path_parts = output_path.split('/')
    for part in path_parts:
        if "\\" in part:
            path_parts.remove(part)
            path_parts += part.split('\\')

    return render_template('detect.html',
                           video_path='/'.join(path_parts),
                           people_count=max_count,
                           is_static=True)

# --- USER Sessions ---

@app.route('/user_dash')
@login_required
def user_dash():
    global stop_feed
    stop_feed = False  # let the video flow

    resource_type = request.form.get('resource_type')
    session['used_resource'] = resource_type
    return render_template('user_dash.html')


@app.route('/user_profile', methods=['GET', 'POST'])
@login_required
def user_profile():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        current_pw = request.form.get('current_password')
        new_pw = request.form.get('new_password')
        confirm_pw = request.form.get('confirm_password')

        user = User.query.get(current_user.id)

        # Change username if needed
        if username and username != user.username:
            if User.query.filter_by(username=username).first():
                flash("Sorry, user already has that username.", "error")
            else:
                user.username = username
                flash("Username updated successfully", "success")

        # Same for email
        if email and email != user.email:
            if User.query.filter_by(email=email).first():
                flash("Email is already taken. Try logging in maybe?", "error")
            else:
                user.email = email
                flash("Email updated", "success")

        if current_pw and new_pw and confirm_pw:
            if check_password_hash(user.password, current_pw):
                if new_pw == confirm_pw:
                    user.password = generate_password_hash(new_pw)
                    flash("New password set successfully", "success")
                else:
                    flash("New passwords did not match", "error")
            else:
                flash("Wrong password. Kindly try again ", "error")

        db.session.commit()
        return redirect(url_for('user_profile'))

    return render_template('user_profile.html', user=current_user)


# --- DETECTION TRIGGER ZONE ---

@app.route('/detect', methods=['GET', 'POST'])
@login_required
def detect():
    global stop_feed, people_count, max_people_count
    people_count = 0
    max_people_count = 0
    stop_feed = False

    if request.method == 'POST':
        res_type = request.form.get('resource_type')
        session['used_resource'] = res_type

        if res_type == 'ip_camera':
            ip_address = request.form.get('ip_address')
            rtsp_url = f"rtsp://{ip_address}:554/stream1"  
            stop_feed = False
            socketio.start_background_task(background_people_count)
            return render_template('detect.html', resource_type='live', people_count=people_count, video_source=rtsp_url)

    # If we didn’t catch POST, fallback 
    fallback_type = request.form.get('resource_type')

    if fallback_type == 'live':
        stop_feed = False
        socketio.start_background_task(background_people_count)
        return render_template('detect.html', resource_type='live', people_count=people_count)
    elif fallback_type == 'image':
        return redirect(url_for('upload_image'))
    elif fallback_type == 'video':
        return redirect(url_for('upload_video'))

    # Just default to the live setup any error or failure
    people_count = 0
    stop_feed = False
    socketio.start_background_task(background_people_count)
    return render_template('detect.html', is_static=False)

# --- AUDIO CHECK ---

@app.route('/check_audio')
def check_audio():
    audio_path = os.path.join(app.config['AUDIO_FOLDER'], 'alert.mp3')

    if os.path.exists(audio_path):
        return f"audio is present {audio_path}"
    else:
        return f"No alert.mp3 found at {audio_path}"


# --- VIDEO FEED ---

@app.route('/video_feed')
def video_feed():
    vid_src = request.args.get('source')
    if vid_src == 'None':
        vid_src = None
    return Response(generate_frames(vid_src),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# --- STOP! and return to user dashboard template for the system ---

@app.route('/stop_feed', methods=['POST'])
@login_required
def stop_feed_func():
    global stop_feed, people_count, max_people_count, heatmap_data

    stop_feed = True
    heatmap_data = []  # Clear heatmap data for the next session
    socketio.emit('feed_stopped')  # Notify client-side that feed has stopped

    import time
    time.sleep(0.5)  

    used = session.get('used_resource', 'Live Feed')
    if used == 'live':
        used = 'Live-Feed'  

    if current_user.is_authenticated:
        log = DetectionLog(
            date=datetime.now().strftime("%Y-%m-%d"),
            time=datetime.now().strftime("%H:%M:%S"),
            used_resource=used,
            total_count=max_people_count,
            user_id=current_user.id
        )
        db.session.add(log)
        db.session.commit()

    # Reset so things start clean next time
    people_count = 0
    max_people_count = 0
    return redirect(url_for('user_dash'))



def background_people_count():
    global stop_feed
    while not stop_feed:
        socketio.emit('people_count', {'count': people_count})
        socketio.sleep(1)  


# --- USER'S PERSONAL DETECTION LOG PAGE ---

@app.route('/user_count')
@login_required
def user_count():
    logs = DetectionLog.query.filter_by(user_id=current_user.id).all()
    return render_template('user_count.html', logs=logs)


# --- Delete a specific log entry from denco database ---

@app.route('/delete_log/<int:log_id>')
@login_required
def delete_log(log_id):
    db.session.execute('PRAGMA foreign_keys = OFF;')  
    try:
        log = DetectionLog.query.get(log_id)
        if log:
            db.session.delete(log)
            db.session.commit()
    except Exception as oops:
        db.session.rollback()
        flash(f"Couldn't delete the log: {oops}", "error")
    finally:
        db.session.execute('PRAGMA foreign_keys = ON;')  # Re-enable foreign key constraints after operation
        db.session.commit()
        reset_auto_increment()

    return redirect(url_for('user_count'))


# --- ADMIN ---

def ensure_admin_exists():
    with app.app_context():
        if User.query.count() == 0:
            # note: this is to initialize our system with a default admin if no users exist
            root_admin = User(
                username='admin',
                email='admin@example.com',
                password=generate_password_hash('admin123'),
                is_admin=True
            )
            db.session.add(root_admin)
            db.session.commit()



if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    with app.app_context():
        db.create_all()
        ensure_admin_exists()

    socketio.run(app, debug=True)
