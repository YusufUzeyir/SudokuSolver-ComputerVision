from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from main import solve_sudoku
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Yükleme klasörünü oluştur
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Dosya kontrolü
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya seçilmedi'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400
    
    # Dosya formatı uygunsa işleme devam et
    if file and allowed_file(file.filename):
        try:
            # Dosyayı oku ve OpenCV formatına çevir
            filestr = file.read()
            npimg = np.frombuffer(filestr, np.uint8)
            image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            
            # Sudoku görüntüsünü çöz
            sonuclar = solve_sudoku(image)
            
            if sonuclar is None:
                return jsonify({'error': 'Sudoku çözülemedi. Lütfen daha net bir görüntü yükleyin.'}), 400
            
            # Tüm aşamaları JSON olarak döndür
            return jsonify(sonuclar)
            
        except Exception as e:
            return jsonify({'error': f'Bir hata oluştu: {str(e)}'}), 500
    
    return jsonify({'error': 'Geçersiz dosya formatı'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 