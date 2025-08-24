import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json

# ===== تحميل النموذج =====
model = load_model("sign_model.h5")
img_size = 64

# ===== قراءة أسماء الحروف من labels.json =====
with open("labels.json", "r", encoding="utf-8") as f:
    categories = json.load(f)
    categories = [categories[str(i)] for i in range(len(categories))]

# تشغيل الكاميرا
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # تحديد مربع أخضر لمكان اليد
    x1, y1, x2, y2 = 100, 100, 300, 300
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    roi = frame[y1:y2, x1:x2]

    # معالجة الصورة: تحويل grayscale + resize + normalize
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (img_size, img_size))
    normalized = resized.reshape(1, img_size, img_size, 1) / 255.0

    # التوقع
    prediction = model.predict(normalized, verbose=0)  # منع الطباعة في كل مرة
    index = np.argmax(prediction)
    label = categories[index]

    # عرض النتيجة أعلى المستطيل مباشرة
    cv2.putText(frame, f"{label}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # إظهار الفيديو
    cv2.imshow("Sign Detection", frame)

    # الخروج عند الضغط على q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
