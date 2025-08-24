import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json

model = load_model("sign_model.h5")
img_size = 64

with open("labels.json", "r", encoding="utf-8") as f:
    categories = json.load(f)
    categories = [categories[str(i)] for i in range(len(categories))]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # adding frame for hand placement
    x1, y1, x2, y2 = 100, 100, 300, 300
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    roi = frame[y1:y2, x1:x2]

    # grayscale & resize & normalize
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (img_size, img_size))
    normalized = resized.reshape(1, img_size, img_size, 1) / 255.0

    # PREDICTION
    prediction = model.predict(normalized, verbose=0)  
    index = np.argmax(prediction)
    label = categories[index]

    # SHOWING RESULT ABOVE SCREEN
    cv2.putText(frame, f"{label}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Sign Detection", frame)

    # for exit press 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
