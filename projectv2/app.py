from flask import Flask, render_template, Response, redirect, url_for, request, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from flask_migrate import Migrate
from ultralytics import YOLO  # Import YOLOv8
import base64
import re


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
    # Detection logs will be accessible via `user.detection_logs`

#     def __repr__(self):
#         return f'<User {self.email}>'

# # Initialize Flask-Admin
# admin = Admin(app, name='Admin Panel', template_mode='bootstrap3')
# admin.add_view(ModelView(User, db.session))

    def __repr__(self):
            return f'<User {self.email}>'

# Define DetectionLog model
class DetectionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(20), nullable=False)
    time = db.Column(db.String(20), nullable=False)
    used_resource = db.Column(db.String(50), nullable=False)
    total_count = db.Column(db.Integer, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  # Foreign Key to user table
    user = db.relationship('User', backref='detection_logs')  # Ensure cascading delete

    def __repr__(self):
        return f'<DetectionLog {self.date} {self.time} {self.used_resource} {self.total_count}>'

class UserView(ModelView):
    column_list = ('id', 'username', 'email', 'password')  # Specify the fields to display
    form_columns = ('username','email', 'password')  # Specify editable fields in the admin form

class DetectionLogView(ModelView):
    column_list = ('id', 'date', 'time', 'used_resource', 'total_count', 'user_id')  # Include user_id
    form_columns = ('date', 'time', 'used_resource', 'total_count', 'user_id')


admin = Admin(app, name='Admin Panel', template_mode='bootstrap4')
admin.add_view(UserView(User, db.session))
admin.add_view(DetectionLogView(DetectionLog, db.session))

def reset_auto_increment():
    db.session.execute('DELETE FROM sqlite_sequence WHERE name="detection_log";')
    db.session.execute('DELETE FROM sqlite_sequence WHERE name="user";')
    db.session.commit()


# Load user function for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

#Landing Page route
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    # your login logic
    return render_template('login_signup.html')


# Login / Signup route
@app.route('/login_signup', methods=['GET', 'POST'])
def login_signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form.get('confirm_password')  # Only for signup
        username = request.form.get('username')

        user = User.query.filter_by(email=email).first()

        # If confirm_password is provided, it indicates a signup request
        if confirm_password:  # Signup logic
            if user:  # Email already exists
                flash('Email already exists! Please log in.', 'signup_error')
                return redirect(url_for('login_signup', form='signup'))
            elif password != confirm_password:  # Passwords don't match
                flash('Passwords do not match!', 'signup_error')
                return redirect(url_for('login_signup', form='signup'))
            else:
                hashed_password = generate_password_hash(password)
                new_user = User(email=email, password=hashed_password, username=username)
                db.session.add(new_user)
                db.session.commit()
                flash('Account created successfully! Please Sign in.', 'signup_success')
                return redirect(url_for('login_signup', form='signup'))

        else:  # Login logic
            if user and check_password_hash(user.password, password):  # Successful login
                login_user(user)
                return redirect(url_for('user_dash'))
            else:  # Invalid credentials
                flash('Wrong Email or Password!', 'login_error')
                return redirect(url_for('login_signup', form='signin'))

    return render_template('login_signup.html')

# User Dashboard route
@app.route('/user_count')
@login_required
def user_count():
    # Fetch detection logs for the current user
    logs = DetectionLog.query.filter_by(user_id=current_user.id).all()
    return render_template('user_count.html', logs=logs)

@app.route('/delete_log/<int:log_id>')
def delete_log(log_id):
    # Start a transaction to disable foreign keys
    db.session.execute('PRAGMA foreign_keys = OFF;')
    
    try:
        # Perform the deletion
        log = DetectionLog.query.get(log_id)
        if log:
            db.session.delete(log)
            db.session.commit()  # Commit the changes after deletion
        
    except Exception as e:
        db.session.rollback()  # In case of error, rollback the transaction
        flash(f"Error deleting record: {e}", 'error')
    finally:
        # Re-enable foreign key constraint
        db.session.execute('PRAGMA foreign_keys = ON;')
        db.session.commit()  # Commit the PRAGMA change
        reset_auto_increment()
    
    return redirect(url_for('user_count'))



#Choice Page(live, image, or vid)
#Change will be made here - below commented
# @app.route('/choice')
# @login_required
# def choice():
#     return render_template('choice.html')

# Detect People/Get Started route
#Change will be made here - below commented
# @app.route('/detect')
# @login_required
# def detect():
#     return render_template('detect.html')

# Logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login_signup'))



                                    # BELOW IS THE DETECTION PHASE

# Change will be made here - below commented
# from flask import Flask, render_template, request, redirect, url_for, Response
# from flask_socketio import SocketIO, emit
# import cv2
# import os
# import numpy as np

# Change will be made here - below commented
# app = Flask(__name__)
# socketio = SocketIO(app)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the folder for uploads exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global variables for people count and control signals
people_count = 0
max_people_count = 0
stop_feed = False

# Load YOLO model for detecting people
def initialize_yolo():
    model = YOLO("yolov8n.pt")  # Load YOLOv8 model (nano version for speed, can use 'yolov8s.pt' or 'yolov8m.pt')
    return model

# Detect people in the frame using YOLO
def detect_people(frame, model):
    global people_count
    results = model(frame)  # Run YOLOv8 inference
    bounding_boxes = []
    confidences = []
    
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, confidence, cls = box.tolist()
            if int(cls) == 0 and confidence > 0.6:  # Class 0 is 'person' in COCO dataset and I reduce the conficdence from 0.6 to 0.3
                bounding_boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                confidences.append(confidence)
    
    people_count = len(bounding_boxes)
    return bounding_boxes, confidences

# Function to generate video frames with detection
def generate_frames():
    global stop_feed, max_people_count
    model = initialize_yolo()
    # List of sources to test (IP Webcam, Extended Camera, Local Webcam)
    video_sources = [
        # "http://172.16.139.219:8080/video"
        "rtsp://172.20.54.250:554/stream1",  # IP Webcam
        1,  # Extended Webcam (USB)
        0   # Local Webcam (Laptop)
    ]

    # Try each source until one works
    cap = None
    for source in video_sources:
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            print(f"Using video source: {source}")
            break
        cap.release()  # Release if not working

    if not cap or not cap.isOpened():
        print("No valid camera source found.")
        return  # Stop execution if no source is available

    
    while True:  # Keep looping until stop_feed is set to True
        if stop_feed:
            break  # Exit immediately

        success, frame = cap.read()
        if not success:
            break
        
        boxes, confidences = detect_people(frame, model)
       

        # Update the maximum count
        max_people_count = max(max_people_count, people_count)
       
        # Draw bounding boxes around detected people
        for i, box in enumerate(boxes):
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {confidences[i]:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)  # Convert frame to JPEG
        frame = buffer.tobytes()  # Convert to bytes for streaming

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Yield frame for streaming

    cap.release()  # Ensure that the webcam is released when done

# Handle Image Upload
@app.route('/upload_image', methods=['POST'])
def upload_image():
    global stop_feed
    stop_feed = True  # Stop any existing live feed
    image = request.files['image']  # Get uploaded image
    if image:
        # Save the uploaded image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(image_path)

          # Redirect to display the image on live_feed page
        return redirect(url_for('display_image', image_path=image_path))
    return redirect(url_for('user_dash'))  # If no image is uploaded, go back to index

# Handle Video Upload
@app.route('/upload_video', methods=['POST'])
def upload_video():
    global stop_feed
    stop_feed = True  # Stop any existing live feed
    video = request.files['video']  # Get uploaded video
    if video:
        # Save the uploaded video
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
        video.save(video_path)

        # Redirect to display the video on live_feed page
        return redirect(url_for('display_video', video_filename=video.filename))
    return redirect(url_for('user_dash'))  # If no video is uploaded, go back to index

# Route to display uploaded image with detection
@app.route('/display_image')
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

    # Log detection data into the database (people count and resource type)
    if current_user.is_authenticated:
        log = DetectionLog(
            date=datetime.now().strftime("%Y-%m-%d"),
            time=datetime.now().strftime("%H:%M:%S"),
            used_resource='Image-Upload',  # Resource type is 'image'
            total_count=people_count,  # The number of people detected in the image
            user_id=current_user.id  # Associate the log with the current user
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
    )  # Display the updated image

# Route to display uploaded video with detection
@app.route('/display_video')
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
        if frame_count % 3 == 0:  # Process every 3rd frame
            boxes, confidences = detect_people(frame, model)
            max_count = max(max_count, len(boxes))

            print(f"Current frame count: {len(boxes)}, Max count so far: {max_count}")

            last_boxes = boxes
            last_confidences = confidences
        
        for i, box in enumerate(last_boxes):
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {last_confidences[i]:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        out.write(frame)
    
    cap.release()
    out.release()

    # Log detection data into the database (people count and resource type)
    if current_user.is_authenticated:
        log = DetectionLog(
            date=datetime.now().strftime("%Y-%m-%d"),
            time=datetime.now().strftime("%H:%M:%S"),
            used_resource='Video-Upload',  # Resource type is 'video'
            total_count=max_count,  # The number of people detected in the video
            user_id=current_user.id  # Associate the log with the current user
        )
        db.session.add(log)
        db.session.commit()

    # Process the path for the processed video
    processed_video_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_video_filename)
    processed_paths = processed_video_path.split('/')
    
    for p in processed_paths:
        if "\\" in p:
            processed_paths.remove(p)
            processed_paths = processed_paths + p.split('\\')
    
    return render_template('detect.html', 
        video_path='/'.join(processed_paths),  # Use processed_paths for video_path
        people_count=max_count,
        is_static=True
    )  # Render video in the detect template

# Route for the landing page
@app.route('/user_dash')
@login_required
def user_dash():
    global stop_feed
    stop_feed = False  # Reset stop feed flag

    resource_type = request.form.get('resource_type')  # "live", "image", or "video"
    session['used_resource'] = resource_type  # Store the choice in session

    return render_template('user_dash.html')

@app.route('/user_profile', methods=['GET', 'POST'])
@login_required
def user_profile():
    if request.method == 'POST':
        # Get form data
        username = request.form.get('username')
        email = request.form.get('email')
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        # Get current user
        user = User.query.get(current_user.id)
        
        # Check if any update is needed
        if username and username != user.username:
            # Check if username already exists
            existing_user = User.query.filter_by(username=username).first()
            if existing_user and existing_user.id != current_user.id:
                flash('Username already exists!', 'error')
            else:
                user.username = username
                flash('Username updated successfully!', 'success')
        
        if email and email != user.email:
            # Check if email already exists
            existing_user = User.query.filter_by(email=email).first()
            if existing_user and existing_user.id != current_user.id:
                flash('Email already exists!', 'error')
            else:
                user.email = email
                flash('Email updated successfully!', 'success')
        
        # Handle password update if current password is provided
        if current_password and new_password and confirm_password:
            if check_password_hash(user.password, current_password):
                if new_password == confirm_password:
                    user.password = generate_password_hash(new_password)
                    flash('Password updated successfully!', 'success')
                else:
                    flash('New passwords do not match!', 'error')
            else:
                flash('Current password is incorrect!', 'error')
        
        # Commit changes to database
        db.session.commit()
        return redirect(url_for('user_profile'))
    
    return render_template('user_profile.html', user=current_user)

# Route for the live feed page
@app.route('/detect', methods=['GET', 'POST'])
@login_required
def detect():
    global stop_feed, people_count, max_people_count

    # Reset people count whenever detect route is accessed
    people_count = 0
    max_people_count = 0
    stop_feed = False  # Ensure live feed starts fresh

    # Get the resource type from the form submission (Live Feed, Image, or Video)
    if request.method == 'POST':
        resource_type = request.form.get('resource_type')
        session['used_resource'] = resource_type  # Store it in the session


    # Get the resource type from the form submission
    resource_type = request.form.get('resource_type')
    
    # Handle the resource type accordingly
    if resource_type == 'live':
        # Start live feed logic
        stop_feed = False  # Ensure the feed starts
        socketio.start_background_task(background_people_count)  # Start background task
        return render_template('detect.html', resource_type='live', people_count=people_count)
    elif resource_type == 'image':
        # Redirect to the image upload route, which already handles everything
        return redirect(url_for('upload_image'))

    elif resource_type == 'video':
        # Redirect to the video upload route, which already handles everything
        return redirect(url_for('upload_video'))

    people_count = 0
    stop_feed = False  # Reset stop feed flag

    # Start the background thread to emit the people count
    socketio.start_background_task(background_people_count)
    return render_template('detect.html',
        is_static=False
    )

# Route for video streaming
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Stop feed and clean up resources
@app.route('/stop_feed', methods=['POST'])
def stop_feed_func():
    global stop_feed, people_count, max_people_count
    stop_feed = True  # Stop the webcam feed
    socketio.emit('feed_stopped')  # Notify the client to stop the feed

     # Get the used resource from session (live, image, or video)
    used_resource = session.get('used_resource', 'Live Feed')  # Default to 'Live Feed' if not set

    # Change 'live' to 'Live-Feed' if the resource is live
    if used_resource == 'live':
        used_resource = 'Live-Feed'

     # Log detection data into the database
    # Save detection log
    if current_user.is_authenticated:
        log = DetectionLog(
            date=datetime.now().strftime("%Y-%m-%d"),
            time=datetime.now().strftime("%H:%M:%S"),
            used_resource=used_resource,  # Or "Image", "Video" based on the session
            total_count=max_people_count,  # The global people count
            user_id=current_user.id  # Associate log with the current user
        )
        db.session.add(log)
        db.session.commit()
    
    # Reset the people count immediately
    people_count = 0  
    max_people_count = 0 

    return redirect(url_for('user_dash'))

# Function to continuously emit the people count to the client
def background_people_count():
    global stop_feed
    while not stop_feed:
        socketio.emit('people_count', {'count': people_count })  # Send people count
        socketio.sleep(1)


# Ensure the database is created when starting the app
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
    with app.app_context():  # Ensure app context is pushed
        db.create_all()  # Create tables if they don't exist
    app.run(debug=True)