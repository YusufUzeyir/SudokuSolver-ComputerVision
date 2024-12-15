import cv2
import numpy as np
from tensorflow.keras.models import load_model
import imutils
from tkinter import Tk, filedialog
from solver import *

sinif_etiketleri = np.arange(0, 10)
model = load_model('model-OCR.h5')
girdi_boyutu = 48

#Seçilen bölgeye perspektif dönüşümü uygular. Dış iskeletin algılanması için kullanılır.
def perspektif_donusturme_uygula(goruntu, kulenin_kose_noktalar, cikti_yuksekligi=900, cikti_genisligi=900):
    pts1 = np.float32([kulenin_kose_noktalar[0], kulenin_kose_noktalar[3], kulenin_kose_noktalar[1], kulenin_kose_noktalar[2]])
    pts2 = np.float32([[0, 0], [cikti_genisligi, 0], [0, cikti_yuksekligi], [cikti_genisligi, cikti_yuksekligi]])

    matris = cv2.getPerspectiveTransform(pts1, pts2)
    donusturulmus_goruntu = cv2.warpPerspective(goruntu, matris, (cikti_genisligi, cikti_yuksekligi))
    return donusturulmus_goruntu

#Perspektif dönüşümünü tersine çevirerek maskeyi orijinal görüntüye geri aktarır.
def ters_perspektif_donusturme(orijinal_goruntu, maskelenmis_sayi, kulenin_kose_noktalar, cikti_yuksekligi=900, cikti_genisligi=900):
    pts1 = np.float32([[0, 0], [cikti_genisligi, 0], [0, cikti_yuksekligi], [cikti_genisligi, cikti_yuksekligi]])
    pts2 = np.float32([kulenin_kose_noktalar[0], kulenin_kose_noktalar[3], kulenin_kose_noktalar[1], kulenin_kose_noktalar[2]])

    matris = cv2.getPerspectiveTransform(pts1, pts2)
    sonuclar = cv2.warpPerspective(maskelenmis_sayi, matris, (orijinal_goruntu.shape[1], orijinal_goruntu.shape[0]))
    return sonuclar

#Verilen görüntüdeki Sudoku tahtasını bulur ve konumunu döndürür.
def sudoku_tahtasini_bul(goruntu):
    gri_goruntu = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
    bulanik_goruntu = cv2.bilateralFilter(gri_goruntu, 13, 20, 20)
    kenarlar = cv2.Canny(bulanik_goruntu, 30, 180)
    
    cv2.imshow("Kenar Algılama", kenarlar)
    kareler_bulundu = cv2.findContours(kenarlar.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    kareler = imutils.grab_contours(kareler_bulundu)

    kareler = sorted(kareler, key=cv2.contourArea, reverse=True)[:15]
    tahta_konumu = None

    for kare in kareler:
        yaklasik = cv2.approxPolyDP(kare, 15, True)
        if len(yaklasik) == 4:
            tahta_konumu = yaklasik
            cv2.drawContours(goruntu, [yaklasik], -1, (0, 255, 0), 2)
            cv2.imshow("kare Bulma", goruntu)
            break
    
    donusturulmus_tahta = perspektif_donusturme_uygula(goruntu, tahta_konumu)
    return donusturulmus_tahta, tahta_konumu

#Sudoku tahtasını 81 ayrı hücreye böler, her hücre bir rakam veya boş bir hücre içerir.
def sudoku_hucrelerini_bol(tahta):
    satirlar = np.vsplit(tahta, 9)
    hucreler = []
    for satir in satirlar:
        sutunlar = np.hsplit(satir, 9)
        for hucre in sutunlar:
            hucre = cv2.resize(hucre, (girdi_boyutu, girdi_boyutu)) / 255.0  #Görüntüdeki piksellerin değerlerini normalize et.
            hucreler.append(hucre)
    cv2.waitKey(1)
    return hucreler

#Sudoku sayılarının her birini, görüntüdeki ilgili hücrelerde gösterir
def sayilari_goruntuye_yerlesitir(goruntu, sayilar, renk=(0, 255, 0)):
    hucre_genisligi = int(goruntu.shape[1] / 9)
    hucre_yuksekligi = int(goruntu.shape[0] / 9)
    for i in range(9):
        for j in range(9):
            if sayilar[(j*9) + i] != 0:
                cv2.putText(goruntu, str(sayilar[(j*9) + i]),
                            (i * hucre_genisligi + int(hucre_genisligi / 2) - int((hucre_genisligi / 4)), int((j + 0.7) * hucre_yuksekligi)),
                            cv2.FONT_HERSHEY_COMPLEX, 2, renk, 2, cv2.LINE_AA)
    return goruntu

# Tkinter dosya seçici kullanımı
Tk().withdraw()  # Ana pencereyi gizler
dosya_yolu = filedialog.askopenfilename(title="Bir Sudoku resmi seçin", filetypes=[("Resim Dosyaları", "*.png;*.jpg;*.jpeg")])
if not dosya_yolu:
    print("Dosya seçilmedi. Program sonlandırılıyor.")
    exit()

girdi_goruntusu = cv2.imread(dosya_yolu)

# Sudoku tahtasını görüntüden çıkar
sudoku_tahtasi, tahta_kose_noktalar = sudoku_tahtasini_bul(girdi_goruntusu)
gri_tahta = cv2.cvtColor(sudoku_tahtasi, cv2.COLOR_BGR2GRAY)
hucreler = sudoku_hucrelerini_bol(gri_tahta)
hucreler = np.array(hucreler).reshape(-1, girdi_boyutu, girdi_boyutu, 1)

# Eğitimli model ile tahmin yap
tahminler = model.predict(hucreler)

tahmin_edilen_sayilar = []
for tahmin in tahminler:
    max_indeks = np.argmax(tahmin)
    tahmin_edilen_sayi = sinif_etiketleri[max_indeks]
    tahmin_edilen_sayilar.append(tahmin_edilen_sayi)

sudoku_izgarasi = np.array(tahmin_edilen_sayilar).astype('uint8').reshape(9, 9)

# Sudoku bulmacasını çöz
try:
    cozulmus_izgara = get_board(sudoku_izgarasi)

    ikili_dizi = np.where(np.array(tahmin_edilen_sayilar) > 0, 0, 1)

    cozulmus_duz_izgara = cozulmus_izgara.flatten() * ikili_dizi

    maske_resmi = np.zeros_like(sudoku_tahtasi)
    cozulmus_tahta_maskesi = sayilari_goruntuye_yerlesitir(maske_resmi, cozulmus_duz_izgara)

    ters_donusturulmus_resim = ters_perspektif_donusturme(girdi_goruntusu, cozulmus_tahta_maskesi, tahta_kose_noktalar)
    birlestirilmis_sonuc = cv2.addWeighted(girdi_goruntusu, 0.7, ters_donusturulmus_resim, 1, 0)
    cv2.imshow("Sonuç", birlestirilmis_sonuc)

except Exception as e:
    print("Çözüm bulunamadı. Model, sayıları yanlış yorumlamış olabilir.")

cv2.waitKey(0)
cv2.destroyAllWindows()
