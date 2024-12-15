from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from solver import get_board
import imutils
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Modeli yükleyin
model = load_model('model-OCR.h5')
girdi_boyutu = 48
sinif_etiketleri = np.arange(0, 10)

# Dosya yükleme için izin verilen dosya uzantıları
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Dosya uzantısı kontrolü
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def perspektif_donusturme_uygula(goruntu, kulenin_kose_noktalar, cikti_yuksekligi=900, cikti_genisligi=900):
    pts1 = np.float32([kulenin_kose_noktalar[0], kulenin_kose_noktalar[3], kulenin_kose_noktalar[1], kulenin_kose_noktalar[2]])
    pts2 = np.float32([[0, 0], [cikti_genisligi, 0], [0, cikti_yuksekligi], [cikti_genisligi, cikti_yuksekligi]])
    matris = cv2.getPerspectiveTransform(pts1, pts2)
    donusturulmus_goruntu = cv2.warpPerspective(goruntu, matris, (cikti_genisligi, cikti_yuksekligi))
    return donusturulmus_goruntu

def sudoku_tahtasini_bul(goruntu):
    gri_goruntu = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
    bulanik_goruntu = cv2.bilateralFilter(gri_goruntu, 13, 20, 20)
    kenarlar = cv2.Canny(bulanik_goruntu, 30, 180)
    kareler_bulundu = cv2.findContours(kenarlar.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    kareler = imutils.grab_contours(kareler_bulundu)
    kareler = sorted(kareler, key=cv2.contourArea, reverse=True)[:15]
    tahta_konumu = None
    for kare in kareler:
        yaklasik = cv2.approxPolyDP(kare, 15, True)
        if len(yaklasik) == 4:
            tahta_konumu = yaklasik
            break
    donusturulmus_tahta = perspektif_donusturme_uygula(goruntu, tahta_konumu)
    return donusturulmus_tahta, tahta_konumu

def sudoku_hucrelerini_bol(tahta):
    satirlar = np.vsplit(tahta, 9)
    hucreler = []
    for satir in satirlar:
        sutunlar = np.hsplit(satir, 9)
        for hucre in sutunlar:
            hucre = cv2.resize(hucre, (girdi_boyutu, girdi_boyutu)) / 255.0
            hucreler.append(hucre)
    return hucreler

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        
        # Resmi oku ve işleyin
        girdi_goruntusu = cv2.imread(os.path.join('uploads', filename))

        # Sudoku tahtasını çıkar
        sudoku_tahtasi, tahta_kose_noktalar = sudoku_tahtasini_bul(girdi_goruntusu)
        gri_tahta = cv2.cvtColor(sudoku_tahtasi, cv2.COLOR_BGR2GRAY)
        hucreler = sudoku_hucrelerini_bol(gri_tahta)
        hucreler = np.array(hucreler).reshape(-1, girdi_boyutu, girdi_boyutu, 1)

        # Model ile tahmin yap
        tahminler = model.predict(hucreler)
        tahmin_edilen_sayilar = []
        for tahmin in tahminler:
            max_indeks = np.argmax(tahmin)
            tahmin_edilen_sayi = sinif_etiketleri[max_indeks]
            tahmin_edilen_sayilar.append(tahmin_edilen_sayi)

        # Sudoku tahtasını çöz
        sudoku_izgarasi = np.array(tahmin_edilen_sayilar).astype('uint8').reshape(9, 9)
        try:
            cozulmus_izgara = get_board(sudoku_izgarasi)
            return jsonify({"solution": cozulmus_izgara.tolist()})
        except Exception as e:
            return f"Çözüm bulunamadı: {str(e)}", 500

    return 'Invalid file format', 400

if __name__ == '__main__':
    app.run(debug=True)
