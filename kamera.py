from ultralytics import YOLO
import cv2

# YOLO modelini yükle
model = YOLO("baret.pt")

# Kameradan görüntü alımı
cap = cv2.VideoCapture(0)

while True:
    # Kameradan bir kare oku
    ret, frame = cap.read()
    if not ret:
        break

    # Model ile tahmin yap
    results = model.predict(source=frame, show=False, conf=0.5)
    helmet_detected = False  # Başlangıçta kask yok olarak kabul edilir

    # Tespit sonuçlarını kontrol et
    for result in results:
        boxes = result.boxes  # Tespit edilen kutular
        if boxes:  # Eğer bir tespit varsa
            for box in boxes:
                # Tespit edilen nesnenin güven oranını kontrol et
                confidence = box.conf[0]
                class_id = int(box.cls[0])  # Nesnenin sınıfı
                if class_id == 0:  # Eğer tespit edilen nesne kask ise
                    helmet_detected = True
                    break  # Kask tespit edildiği için döngüyü kırıyoruz

    # Mesaj yazdır
    if helmet_detected:
        cv2.putText(frame, "Baret takmak onemlidir!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Bareti tak!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Sonuçları ekranda göster
    cv2.imshow('frame', frame)

    # 'd' tuşuna basıldığında çıkış yap
    if cv2.waitKey(20) & 0xFF == ord('d'):
        break

# Kamerayı serbest bırak ve tüm pencereleri kapat
cap.release()
cv2.destroyAllWindows()
