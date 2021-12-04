import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory,flash 
from werkzeug.utils import secure_filename
import cv2
import numpy as np

import detector

UPLOAD_FOLDER ='static/uploads/'
DOWNLOAD_FOLDER = 'static/downloads/'
ALLOWED_EXTENSIONS = {'jpg', 'png','jpeg'}
app = Flask(__name__, static_url_path="/static")

app.config['SECRET_KEY'] = 'opencv'  
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
# limit upload size upto 6mb
app.config['MAX_CONTENT_LENGTH'] = 6 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods = ["POST", "GET"])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No attach file in request')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            prediction = detector.detection(os.path.join(UPLOAD_FOLDER, filename))
            print(prediction)
            data = {
                "uploaded_image" : 'static/uploads/'+filename,
                "class" : prediction 
            }
            return render_template('index.html', data = data)
    return render_template('index.html')
        

if __name__ == '__main__':
    app.run(debug = True)