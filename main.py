# Uyarı mesajlarını gizle
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow uyarılarını gizle
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # oneDNN uyarılarını kapat
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # Sadece hataları göster

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import imutils
from solver import *
import base64

sinif_etiketleri = np.arange(0, 10)
model = load_model('model1-OCR.h5')
girdi_boyutu = 48

def perspektif_donusturme_uygula(goruntu, kulenin_kose_noktalar, cikti_yuksekligi=900, cikti_genisligi=900):
    """
    Seçilen bölgeye perspektif dönüşümü uygular.
    Sudoku tahtasını düz bir görünüme çevirmek için kullanılır.
    """
    pts1 = np.float32([kulenin_kose_noktalar[0], kulenin_kose_noktalar[3], 
                       kulenin_kose_noktalar[1], kulenin_kose_noktalar[2]])
    pts2 = np.float32([[0, 0], [cikti_genisligi, 0], 
                       [0, cikti_yuksekligi], [cikti_genisligi, cikti_yuksekligi]])

    matris = cv2.getPerspectiveTransform(pts1, pts2)
    donusturulmus_goruntu = cv2.warpPerspective(goruntu, matris, (cikti_genisligi, cikti_yuksekligi))
    return donusturulmus_goruntu

def ters_perspektif_donusturme(orijinal_goruntu, maskelenmis_sayi, kulenin_kose_noktalar, 
                              cikti_yuksekligi=900, cikti_genisligi=900):
    """
    Perspektif dönüşümünü tersine çevirerek çözümü orijinal görüntüye yerleştirir.
    Çözümü orijinal fotoğrafın üzerine yerleştirmek için kullanılır.
    """
    pts1 = np.float32([[0, 0], [cikti_genisligi, 0], 
                       [0, cikti_yuksekligi], [cikti_genisligi, cikti_yuksekligi]])
    pts2 = np.float32([kulenin_kose_noktalar[0], kulenin_kose_noktalar[3], 
                       kulenin_kose_noktalar[1], kulenin_kose_noktalar[2]])

    matris = cv2.getPerspectiveTransform(pts1, pts2)
    sonuclar = cv2.warpPerspective(maskelenmis_sayi, matris, 
                                  (orijinal_goruntu.shape[1], orijinal_goruntu.shape[0]))
    return sonuclar

def sudoku_tahtasini_bul(goruntu):
    """
    Verilen görüntüdeki Sudoku tahtasını tespit eder.
    1. Görüntüyü gri tonlamaya çevirir
    2. Kenarları belirginleştirir
    3. En büyük kare şekli bulur (Sudoku tahtası)
    """
    gri_goruntu = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
    bulanik_goruntu = cv2.bilateralFilter(gri_goruntu, 13, 20, 20)
    kenarlar = cv2.Canny(bulanik_goruntu, 30, 180)
    
    kareler_bulundu = cv2.findContours(kenarlar.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    kareler = imutils.grab_contours(kareler_bulundu)

    # En büyük kareden başlayarak Sudoku tahtasını bul
    kareler = sorted(kareler, key=cv2.contourArea, reverse=True)[:15]
    tahta_konumu = None

    for kare in kareler:
        yaklasik = cv2.approxPolyDP(kare, 15, True)
        if len(yaklasik) == 4:  # Dört köşesi olan şekli bul
            tahta_konumu = yaklasik
            break
    
    if tahta_konumu is None:
        raise ValueError("Sudoku tahtası bulunamadı!")
    
    donusturulmus_tahta = perspektif_donusturme_uygula(goruntu, tahta_konumu)
    return donusturulmus_tahta, tahta_konumu

def sudoku_hucrelerini_bol(tahta):
    """
    Sudoku tahtasını 81 ayrı hücreye böler.
    Her hücreyi model için uygun boyuta getirir.
    """
    satirlar = np.vsplit(tahta, 9)  # Dikey olarak 9'a böl
    hucreler = []
    for satir in satirlar:
        sutunlar = np.hsplit(satir, 9)  # Yatay olarak 9'a böl
        for hucre in sutunlar:
            hucre = cv2.resize(hucre, (girdi_boyutu, girdi_boyutu))  # Model için yeniden boyutlandır
            hucre = cv2.cvtColor(hucre, cv2.COLOR_BGR2GRAY)  # Gri tonlamaya çevir
            hucre = hucre.astype('float32') / 255.0  # Normalize et
            hucreler.append(hucre)
    return hucreler

def sayilari_goruntuye_yerlesitir(goruntu, sayilar, renk=(0, 255, 0)):
    """
    Çözülen Sudoku sayılarını görüntüye yerleştirir.
    Her sayıyı doğru konuma ve boyutta yerleştirir.
    """
    hucre_genisligi = int(goruntu.shape[1] / 9)
    hucre_yuksekligi = int(goruntu.shape[0] / 9)
    for i in range(9):
        for j in range(9):
            if sayilar[(j*9) + i] != 0:  # Sadece çözülen sayıları yerleştir
                cv2.putText(goruntu, str(sayilar[(j*9) + i]),
                           (i * hucre_genisligi + int(hucre_genisligi / 2) - int((hucre_genisligi / 4)),
                            int((j + 0.7) * hucre_yuksekligi)),
                           cv2.FONT_HERSHEY_COMPLEX, 2, renk, 2, cv2.LINE_AA)
    return goruntu

def solve_sudoku(image):
    """
    Ana fonksiyon: Sudoku görüntüsünü alır, işler ve çözümü döndürür.
    İşlem adımları:
    1. Sudoku tahtasını tespit et
    2. Rakamları tanı
    3. Sudoku'yu çöz
    4. Çözümü orijinal görüntüye yerleştir
    """
    try:
        # Sudoku tahtasını bul ve perspektif düzelt
        gri_goruntu = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bulanik_goruntu = cv2.bilateralFilter(gri_goruntu, 13, 20, 20)
        kenarlar = cv2.Canny(bulanik_goruntu, 30, 180)
        
        # Kenarları renkli göstermek için
        kenarlar_renkli = cv2.cvtColor(kenarlar, cv2.COLOR_GRAY2BGR)
        
        kareler_bulundu = cv2.findContours(kenarlar.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        kareler = imutils.grab_contours(kareler_bulundu)
        kareler = sorted(kareler, key=cv2.contourArea, reverse=True)[:15]
        
        # Konturları çiz
        konturlu_goruntu = image.copy()
        cv2.drawContours(konturlu_goruntu, kareler, -1, (0, 255, 0), 3)
        
        tahta_konumu = None
        for kare in kareler:
            yaklasik = cv2.approxPolyDP(kare, 15, True)
            if len(yaklasik) == 4:
                tahta_konumu = yaklasik
                break
        
        if tahta_konumu is None:
            raise ValueError("Sudoku tahtası bulunamadı!")
        
        sudoku_tahtasi = perspektif_donusturme_uygula(image, tahta_konumu)
        
        # Hücreleri böl ve rakamları tanı
        hucreler = sudoku_hucrelerini_bol(sudoku_tahtasi)
        hucreler = np.array(hucreler).reshape(-1, girdi_boyutu, girdi_boyutu, 1)
        
        # Yapay zeka modeli ile rakamları tahmin et
        tahminler = model.predict(hucreler, verbose=0)
        tahmin_edilen_sayilar = []
        for tahmin in tahminler:
            max_indeks = np.argmax(tahmin)
            tahmin_edilen_sayi = sinif_etiketleri[max_indeks]
            tahmin_edilen_sayilar.append(tahmin_edilen_sayi)
        
        # Sudoku ızgarasını oluştur ve çöz
        sudoku_izgarasi = np.array(tahmin_edilen_sayilar).astype('uint8').reshape(9, 9)
        cozulmus_izgara = get_board(sudoku_izgarasi)
        
        # Çözümü görselleştir
        ikili_dizi = np.where(np.array(tahmin_edilen_sayilar) > 0, 0, 1)
        cozulmus_duz_izgara = cozulmus_izgara.flatten() * ikili_dizi
        
        # Çözümü orijinal görüntüye yerleştir
        maske_resmi = np.zeros_like(sudoku_tahtasi)
        cozulmus_tahta_maskesi = sayilari_goruntuye_yerlesitir(maske_resmi, cozulmus_duz_izgara)
        ters_donusturulmus_resim = ters_perspektif_donusturme(image, cozulmus_tahta_maskesi, tahta_konumu)
        sonuc = cv2.addWeighted(image, 0.7, ters_donusturulmus_resim, 1, 0)
        
        # Tüm aşamaları base64'e çevir
        _, buffer1 = cv2.imencode('.png', konturlu_goruntu)
        _, buffer2 = cv2.imencode('.png', kenarlar_renkli)
        _, buffer3 = cv2.imencode('.png', sonuc)
        
        return {
            'kareler': base64.b64encode(buffer1).decode(),
            'kenarlar': base64.b64encode(buffer2).decode(),
            'sonuc': base64.b64encode(buffer3).decode()
        }
        
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None
