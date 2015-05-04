import os
import random

from PIL import Image
from flask import Flask
from flask import request, render_template, jsonify, redirect, url_for, request
from coin import utils, segmenter

app = Flask(__name__)
app.config.from_object(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'jpeg', 'gif', 'PNG', 'JPEG', 'GIF'])
eng = utils.getEngine()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], "mostImportantFile.%s" % file.filename.rsplit('.', 1)[1])
            file.save(filename)
            return jsonify({"success":True})
        print("failed")
        return jsonify({"fail":True})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/process', methods=['GET'])
def process():
    if request.method == 'GET':
        return jsonify(segmenter.processImageforJSON('uploads/mostImportantFile.jpg', eng))

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
