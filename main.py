import time
import os
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response
import subprocess
import pytesseract
from PIL import Image
import logging
from celery import Celery, signals, group
import sqlite3
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import ssl
import argparse
from itertools import islice
import cv2
import numpy as np
import dbus
import re

# Configuration
SCREENSHOT_INTERVAL = 5 * 60  # 5 minutes
SCREENSHOT_DIR = "static/screenshots"
DATABASE = "screenshots.db"

# Ensure screenshot directory exists
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

# Celery configuration
app.config['CELERY_BROKER_URL'] = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
app.config['CELERY_RESULT_BACKEND'] = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
app.config['CELERYD_CONCURRENCY'] = 2  # Limit to 2 concurrent workers

# Initialize Celery
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# Database initialization function
def init_db():
    with sqlite3.connect(DATABASE) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS screenshots
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         filename TEXT NOT NULL,
                         timestamp TEXT NOT NULL,
                         ocr_text TEXT,
                         tags TEXT)''')

init_db()

# Function to ensure NLTK data is downloaded
def ensure_nltk_data():
    try:
        _create_unverified_https_context = ssl._create_unverified_https_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

ensure_nltk_data()

# Function to get existing words from the database
def get_existing_words():
    with sqlite3.connect(DATABASE) as conn:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT tags FROM screenshots WHERE tags IS NOT NULL")
        all_tags = cur.fetchall()
    existing_words = set()
    for tags in all_tags:
        if tags[0]:
            existing_words.update(json.loads(tags[0]))
    return existing_words

# Function to generate tags from OCR text
def generate_tags(ocr_text):
    tokens = word_tokenize(ocr_text.lower())
    stop_words = set(stopwords.words('english'))
    existing_words = get_existing_words()
    tags = [word for word in tokens if word.isalnum() and word not in stop_words and word in existing_words]
    return list(set(tags))[:5]  # Remove duplicates and limit to 5 tags

# Global variable to store the OCR engine
ocr_engine = None

# Function to initialize the OCR engine
def initialize_ocr_engine():
    global ocr_engine
    ocr_engine = pytesseract
    print("Tesseract OCR initialized successfully.")

# Celery task to process screenshots
@celery.task
def process_screenshot(image_path):
    try:
        logger.info(f"Performing OCR on {image_path}")
        ocr_text = multi_pass_ocr(image_path)
        tags = generate_tags(ocr_text)
        logger.info(f"OCR completed for {image_path}")
        
        with sqlite3.connect(DATABASE) as conn:
            conn.execute("UPDATE screenshots SET ocr_text = ?, tags = ? WHERE filename = ?",
                         (ocr_text, json.dumps(tags), image_path))
        conn.commit()
        
        return ocr_text
    except Exception as e:
        logger.error(f"Error performing OCR on {image_path}: {e}")
        return ""

# Function to preprocess the image for OCR
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    # Increase resolution
    scaled = cv2.resize(denoised, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return scaled

# Function to isolate text regions in the image
def fast_isolate_text_regions(img):
    edges = cv2.Canny(img, 100, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dilated = cv2.dilate(edges, kernel, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(img.shape, dtype=np.uint8)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / float(h)
        if 100 < area < 50000 and 0.1 < aspect_ratio < 10:
            cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
    result = cv2.bitwise_and(img, mask)
    return result

# Function to perform OCR on an image
def perform_ocr(image):
    custom_config = r'--oem 3 --psm 6 -l eng'  # Assume English language
    text = pytesseract.image_to_string(Image.fromarray(image), config=custom_config)
    return clean_ocr_text(text)

# Function to clean OCR text
def clean_ocr_text(text):
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable())
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove single characters (often errors)
    text = re.sub(r'\b\w\b', '', text)
    return text

# Function to perform multiple OCR passes
def multi_pass_ocr(image_path):
    img = cv2.imread(image_path)
    results = []

    # Original image
    results.append(perform_ocr(img))

    # Preprocessed image
    preprocessed = preprocess_image(image_path)
    results.append(perform_ocr(preprocessed))

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results.append(perform_ocr(gray))

    # Thresholded
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results.append(perform_ocr(thresh))

    # Combine results
    combined = ' '.join(set(' '.join(results).split()))
    return clean_ocr_text(combined)

# Function to get the session type (X11 or Wayland)
def get_session_type():
    try:
        return subprocess.check_output(["echo $XDG_SESSION_TYPE"], shell=True).decode().strip()
    except subprocess.CalledProcessError:
        logger.warning("Failed to determine session type. Assuming X11.")
        return "x11"

# Function to check if the system is idle
def is_system_idle(idle_time_minutes=5):
    print("Checking if system is idle")
    idle_time_ms = idle_time_minutes * 60 * 1000
    session_type = get_session_type()

    if session_type == "x11":
        try:
            idle_time = int(subprocess.check_output(['xprintidle']).decode().strip())
            return idle_time > idle_time_ms
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("Failed to check idle state with xprintidle. Make sure it's installed.")
            return False
    elif session_type == "wayland":
        try:
            bus = dbus.SessionBus()
            screensaver_proxy = bus.get_object('org.freedesktop.ScreenSaver', '/org/freedesktop/ScreenSaver')
            screensaver_interface = dbus.Interface(screensaver_proxy, dbus_interface='org.freedesktop.ScreenSaver')
            idle_time = screensaver_interface.GetSessionIdleTime()
            return idle_time > idle_time_ms
        except dbus.exceptions.DBusException as e:
            logger.error(f"Failed to check idle state via DBus: {e}")
            return False
    else:
        logger.warning(f"Unknown session type: {session_type}. Assuming not idle.")
        return False

# Function to take screenshots
def take_screenshot():
    while True:
        if not is_system_idle():  # Only take screenshot if system is not idle
            print("System is not idle, taking screenshot")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.abspath(f"{SCREENSHOT_DIR}/screenshot_{timestamp}.png")
            try:
                subprocess.run([
                    "spectacle",
                    "-b",  # background mode
                    "-n",  # no notification
                    "-o", filename,  # output file
                    "-f"  # full screen
                ], check=True)
                logger.info(f"Screenshot saved: {filename}")
                
                # Store screenshot info in database
                with sqlite3.connect(DATABASE) as conn:
                    conn.execute("INSERT INTO screenshots (filename, timestamp) VALUES (?, ?)",
                                 (filename, timestamp))
                
                # Trigger async OCR task
                process_screenshot.delay(filename)
            except subprocess.CalledProcessError as e:  
                logger.error(f"Error taking screenshot: {e}")
        else:
            logger.info("System is idle, skipping screenshot")
        time.sleep(SCREENSHOT_INTERVAL)

# Flask route for the main page
@app.route('/')
def index():
    with sqlite3.connect(DATABASE) as conn:
        cur = conn.cursor()
        cur.execute("SELECT filename, timestamp, ocr_text, tags FROM screenshots ORDER BY timestamp DESC")
        screenshots = cur.fetchall()
    screenshots = [{'filename': os.path.basename(s[0]), 'timestamp': s[1], 'formatted_timestamp': format_timestamp(s[1]), 'ocr_status': bool(s[2]), 'tags': json.loads(s[3]) if s[3] else []} for s in screenshots]
    return render_template('index.html', screenshots=screenshots)

# Flask route for search functionality
@app.route('/search', methods=['POST'])
def search():   
    query = request.form.get('query', '').lower()
    with sqlite3.connect(DATABASE) as conn:
        cur = conn.cursor()
        cur.execute("SELECT filename, timestamp, ocr_text, tags FROM screenshots WHERE LOWER(ocr_text) LIKE ?", (f'%{query}%',))
        results = cur.fetchall()
    return jsonify([{'filename': os.path.basename(r[0]), 'timestamp': r[1], 'formatted_timestamp': format_timestamp(r[1]), 'ocr_status': bool(r[2]), 'tags': json.loads(r[3]) if r[3] else []} for r in results])

# Function to batch process screenshots
def batch_process_screenshots(batch_size=5):
    with sqlite3.connect(DATABASE) as conn:
        cur = conn.cursor()
        cur.execute("SELECT filename FROM screenshots WHERE ocr_text IS NULL")
        screenshots = cur.fetchall()
    
    def chunks(data, size):
        it = iter(data)
        return iter(lambda: tuple(islice(it, size)), ())
    
    for batch in chunks(screenshots, batch_size):
        group(process_screenshot.s(screenshot[0]) for screenshot in batch)().get()

# Flask route to trigger OCR for all unprocessed images
@app.route('/ocr-all', methods=['POST'])
def ocr_all():
    batch_process_screenshots.delay()
    return jsonify({"message": "OCR started for all unprocessed images in batches."})

# Celery task for batch processing screenshots
@celery.task
def batch_process_screenshots(batch_size=5):
    with sqlite3.connect(DATABASE) as conn:
        cur = conn.cursor()
        cur.execute("SELECT filename FROM screenshots WHERE ocr_text IS NULL")
        screenshots = cur.fetchall()
    
    def chunks(data, size):
        it = iter(data)
        return iter(lambda: tuple(islice(it, size)), ())
    
    for batch in chunks(screenshots, batch_size):
        group(process_screenshot.s(screenshot[0]) for screenshot in batch)().get()

# Flask route to delete all screenshots
@app.route('/delete-all', methods=['POST'])
def delete_all():
    with sqlite3.connect(DATABASE) as conn:
        cur = conn.cursor()
        cur.execute("SELECT filename FROM screenshots")
        screenshots = cur.fetchall()
        for screenshot in screenshots:
            os.remove(screenshot[0])
        cur.execute("DELETE FROM screenshots")
    return jsonify({"message": "All screenshots deleted."})

# Flask route to set screenshot interval
@app.route('/set-interval', methods=['POST'])
def set_interval():
    interval = request.form.get('interval', type=int)
    if interval:
        global SCREENSHOT_INTERVAL
        SCREENSHOT_INTERVAL = interval
        return jsonify({"message": f"Screenshot interval set to {interval} seconds."})
    return jsonify({"message": "Invalid interval."})

# Flask route for status updates
@app.route('/status-updates')
def status_updates():
    def generate():
        with sqlite3.connect(DATABASE) as conn:
            cur = conn.cursor()
            last_id = 0
            while True:
                cur.execute("SELECT id, filename, timestamp, ocr_text, tags FROM screenshots WHERE id > ? ORDER BY id", (last_id,))
                results = cur.fetchall()
                for row in results:
                    last_id = row[0]
                    status = "Analyzed" if row[3] else "Not yet analyzed"
                    data = {
                        'id': row[0],
                        'filename': os.path.basename(row[1]),
                        'timestamp': row[2],
                        'status': status,
                        'is_new': last_id == row[0],
                        'tags': json.loads(row[4]) if row[4] else []
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                time.sleep(1)

    return Response(generate(), mimetype='text/event-stream')

# Flask route to delete all screenshots and reset the database
@app.route('/delete-all-and-reset-db', methods=['POST'])
def delete_all_and_reset_db():
    try:
        # Delete all screenshot files
        for filename in os.listdir(SCREENSHOT_DIR):
            file_path = os.path.join(SCREENSHOT_DIR, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        
        # Reset the database
        with sqlite3.connect(DATABASE) as conn:
            conn.execute("DELETE FROM screenshots")
            conn.execute("DELETE FROM sqlite_sequence WHERE name='screenshots'")
        
        return jsonify({"message": "All screenshots deleted and database reset successfully."})
    except Exception as e:
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500

# Flask route to filter screenshots by tag
@app.route('/filter-by-tag/<tag>')
def filter_by_tag(tag):
    with sqlite3.connect(DATABASE) as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM screenshots WHERE tags LIKE ?", (f'%"{tag}"%',))
        screenshots = cur.fetchall()
    return jsonify([{
        'id': s[0],
        'filename': s[1],
        'timestamp': s[2],
        'ocr_text': s[3],
        'tags': json.loads(s[4]) if s[4] else []
    } for s in screenshots])

# Flask route to get all unique tags
@app.route('/get-all-tags')
def get_all_tags():
    with sqlite3.connect(DATABASE) as conn:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT tags FROM screenshots WHERE tags IS NOT NULL")
        all_tags = cur.fetchall()
    unique_tags = set()
    for tags in all_tags:
        if tags[0]:
            unique_tags.update(json.loads(tags[0]))
    return jsonify(list(unique_tags))

# Flask route to update tags for a screenshot
@app.route('/update_tags', methods=['POST'])
def update_tags():
    data = request.json
    filename = data['filename']
    new_tags = data['tags']
    this_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(this_dir, SCREENSHOT_DIR, filename) 
    print("Updating tags for:", filepath)
    try:
        with sqlite3.connect(DATABASE) as conn:
            cur = conn.cursor()
            cur.execute("UPDATE screenshots SET tags = ? WHERE filename = ?", 
                        (json.dumps(new_tags), filepath))
            conn.commit()
        return jsonify({"success": True, "message": "Tags updated successfully"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# Flask route to get information about a specific screenshot
@app.route('/get_screenshot_info/<filename>')
def get_screenshot_info(filename):
    with sqlite3.connect(DATABASE) as conn:
        cur = conn.cursor()
        this_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(this_dir, SCREENSHOT_DIR, filename) 
        print("Getting info for:", filepath) 
        cur.execute("SELECT timestamp, ocr_text, tags FROM screenshots WHERE filename = ?", (filepath,))
        result = cur.fetchone()
    
    if result:
        return jsonify({
            "timestamp": result[0],
            "ocr_text": result[1],
            "tags": json.loads(result[2]) if result[2] else []
        })
    else:
        return jsonify({"error": "Screenshot not found"}), 404

# Function to format timestamp
def format_timestamp(timestamp):
    try:
        dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return "Invalid Date"

# Main function
def main():
    # Start screenshot thread
    screenshot_thread = threading.Thread(target=take_screenshot, daemon=True)
    screenshot_thread.start()
    
    app.run(debug=True, use_reloader=False)

if __name__ == "__main__":
    initialize_ocr_engine()
    ensure_nltk_data()  # Ensure NLTK data is available before starting the app
    main()