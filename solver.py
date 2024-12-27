import cv2
import numpy as np
import tensorflow as tf
import imutils

# Eğitilmiş OCR modelini yükle
model = tf.keras.models.load_model('model1-OCR.h5')
INPUT_SIZE = 48  # Model girdi boyutu

def find_empty(board):
    """
    Sudoku tahtasında boş (0 değerli) bir hücre arar.
    Args:
        board: 9x9 Sudoku tahtası
    Returns:
        (satır, sütun) koordinatları veya None
    """
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 0:
                return (i, j)  # satır, sütun
    return None

def valid(board, num, pos):
    """
    Verilen sayının belirtilen konuma yerleştirilebilir olup olmadığını kontrol eder.
    Sudoku kurallarına göre satır, sütun ve 3x3'lük kutu içinde aynı sayı olmamalıdır.
    
    Args:
        board: 9x9 Sudoku tahtası
        num: Yerleştirilecek sayı
        pos: (satır, sütun) koordinatları
    Returns:
        bool: Sayının yerleştirilebilir olup olmadığı
    """
    # Satır kontrolü - aynı satırda aynı sayı var mı?
    for i in range(len(board[0])):
        if board[pos[0]][i] == num and pos[1] != i:
            return False

    # Sütun kontrolü - aynı sütunda aynı sayı var mı?
    for i in range(len(board)):
        if board[i][pos[1]] == num and pos[0] != i:
            return False

    # 3x3'lük kutu kontrolü
    box_x = pos[1] // 3  # Kutunun x koordinatı
    box_y = pos[0] // 3  # Kutunun y koordinatı

    # 3x3'lük kutu içinde aynı sayı var mı?
    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if board[i][j] == num and (i,j) != pos:
                return False

    return True

def solve(board):
    """
    Sudoku tahtasını recursive backtracking algoritması ile çözer.
    
    Args:
        board: 9x9 Sudoku tahtası
    Returns:
        bool: Çözümün bulunup bulunmadığı
    """
    # Boş hücre bul
    find = find_empty(board)
    if not find:  # Boş hücre kalmadıysa çözüm tamamlanmıştır
        return True
    else:
        row, col = find

    # 1'den 9'a kadar sayıları dene
    for i in range(1,10):
        if valid(board, i, (row, col)):  # Sayı yerleştirilebilir mi?
            board[row][col] = i  # Sayıyı yerleştir

            if solve(board):  # Recursive olarak devam et
                return True
            board[row][col] = 0  # Çözüm bulunamadıysa geri al
    return False

def get_board(bo):
    """
    Sudoku tahtasını çözer ve sonucu döndürür.
    
    Args:
        bo: 9x9 çözülmemiş Sudoku tahtası
    Returns:
        9x9 çözülmüş Sudoku tahtası
    Raises:
        ValueError: Sudoku çözülemezse
    """
    if solve(bo):
        return bo
    else:
        raise ValueError("Geçersiz Sudoku tahtası!")

def preprocess_image(image):
    """
    Görüntüyü işleyerek Sudoku karesini bulur ve perspektif düzeltmesi yapar.
    
    İşlem adımları:
    1. Görüntüyü gri tonlamaya çevir
    2. Gürültüyü azalt
    3. Kenarları tespit et
    4. Sudoku karesini bul
    5. Perspektif düzeltmesi yap
    
    Args:
        image: Orijinal görüntü
    Returns:
        warped: Düzeltilmiş Sudoku karesi
        matrix: Perspektif dönüşüm matrisi
        pts1: Orijinal köşe noktaları
    """
    # Görüntüyü gri tonlamaya dönüştür
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Görüntüyü bulanıklaştır (gürültüyü azalt)
    blur = cv2.bilateralFilter(gray, 13, 20, 20)
    
    # Kenar tespiti
    edges = cv2.Canny(blur, 30, 180)
    
    # Konturları bul
    contours = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    
    # En büyük konturları al (Sudoku karesi en büyük kare olmalı)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    sudoku_contour = None
    
    # Dört köşeli şekli bul (Sudoku karesi)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 15, True)
        if len(approx) == 4:
            sudoku_contour = approx
            break
    
    if sudoku_contour is None:
        raise ValueError("Sudoku karesini bulamadım!")
    
    # Perspektif dönüşümü için köşe noktaları düzenle
    pts = np.float32([sudoku_contour[0][0], sudoku_contour[1][0], 
                      sudoku_contour[2][0], sudoku_contour[3][0]])
    pts = pts[pts[:, 1].argsort()]  # Y koordinatına göre sırala
    top = pts[:2][pts[:2, 0].argsort()]  # Üst noktalar
    bottom = pts[2:][pts[2:, 0].argsort()]  # Alt noktalar
    pts1 = np.float32([top[0], top[1], bottom[0], bottom[1]])
    pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])  # Hedef noktalar
    
    # Perspektif dönüşümü uygula
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(gray, matrix, (450, 450))
    
    return warped, matrix, pts1

def get_digits(warped):
    """
    Düzeltilmiş Sudoku karesinden rakamları tanır.
    
    İşlem adımları:
    1. Görüntüyü iyileştir
    2. Her hücreyi tek tek işle
    3. Rakamları yapay zeka modeli ile tanı
    
    Args:
        warped: Düzeltilmiş Sudoku karesi
    Returns:
        board: 9x9 sayısal Sudoku tahtası
    """
    board = np.zeros((9, 9), dtype=np.int32)
    cell_size = warped.shape[0] // 9  # Her hücrenin boyutu
    
    # Görüntüyü iyileştir (rakamları belirginleştir)
    warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Her hücreyi işle
    for i in range(9):
        for j in range(9):
            # Hücreyi kes
            cell = warped[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            
            # Hücre içindeki rakamı bul
            contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:  # Rakam yoksa
                board[i][j] = 0
                continue
                
            # En büyük konturu al (rakam en büyük kontur olmalı)
            max_area = 0
            digit_contour = None
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area and area > (cell_size * cell_size * 0.01):
                    max_area = area
                    digit_contour = contour
            
            if digit_contour is None:  # Geçerli rakam bulunamadıysa
                board[i][j] = 0
                continue
                
            # Rakam bölgesini kes
            x, y, w, h = cv2.boundingRect(digit_contour)
            digit = cell[y:y+h, x:x+w]
            
            # Rakamı model için hazırla
            digit = cv2.resize(digit, (INPUT_SIZE, INPUT_SIZE))
            digit = digit.astype('float32') / 255.0  # Normalize et
            digit = np.expand_dims(digit, axis=-1)  # Kanal ekle
            digit = np.expand_dims(digit, axis=0)  # Batch boyutu ekle
            
            # Yapay zeka ile rakamı tanı
            prediction = model.predict(digit, verbose=0)
            predicted_digit = np.argmax(prediction[0])  # En yüksek olasılıklı rakam
            confidence = np.max(prediction[0])  # Tahmin güveni
            
            # Güven yeterli ise rakamı kaydet
            if confidence > 0.5:
                board[i][j] = predicted_digit
            else:
                board[i][j] = 0
    
    return board

def solve_sudoku(image):
    """
    Ana fonksiyon: Sudoku görüntüsünü alır, işler ve çözümü döndürür.
    
    İşlem adımları:
    1. Görüntüyü işle ve Sudoku karesini bul
    2. Rakamları tanı
    3. Sudoku bulmacasını çöz
    
    Args:
        image: Orijinal Sudoku görüntüsü
    Returns:
        solution: Çözülmüş Sudoku tahtası
        transform_matrix: Perspektif dönüşüm matrisi
        original_points: Orijinal köşe noktaları
    """
    try:
        print("Görüntü işleniyor...")
        # Görüntüyü işle ve Sudoku karesini bul
        warped, transform_matrix, original_points = preprocess_image(image)
        
        print("Rakamlar tanınıyor...")
        # Yapay zeka ile rakamları tanı
        board = get_digits(warped)
        
        print("Sudoku çözülüyor...")
        # Backtracking algoritması ile Sudoku'yu çöz
        solution = get_board(board)
        
        print("Çözüm başarılı!")
        return solution, transform_matrix, original_points
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, None, None


