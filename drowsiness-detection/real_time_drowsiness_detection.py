import cv2
import numpy as np
import tensorflow as tf

# Tải mô hình đã huấn luyện
model = tf.keras.models.load_model('my_model.keras')

def preprocess_eye_image(eye_image):
    eye_image = cv2.resize(eye_image, (48, 48))
    if len(eye_image.shape) == 2:  # Chuyển đổi từ grayscale sang RGB nếu cần
        eye_image = cv2.cvtColor(eye_image, cv2.COLOR_GRAY2RGB)
    eye_image = np.expand_dims(eye_image, axis=0)  # Thêm batch dimension
    eye_image = eye_image / 255.0  # Chuẩn hóa ảnh về [0, 1]
    return eye_image

def predict_eye_state(eye_image):
    eye_image = preprocess_eye_image(eye_image)
    pred = model.predict(eye_image)
    return pred[0][0]  # Trả về dự đoán trạng thái mắt

# Phát hiện khuôn mặt và mắt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Khởi động camera
cap = cv2.VideoCapture(0)
closed_frames = 0  # Biến đếm số khung hình mắt nhắm

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            eye = roi_gray[ey:ey+eh, ex:ex+ew]
            state = predict_eye_state(eye)  # Dự đoán trạng thái mắt

            label = "Open" if state >= 0.5 else "Closed"
            color = (0, 255, 0) if state >= 0.5 else (0, 0, 255)
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), color, 2)
            cv2.putText(frame, label, (x + ex, y + ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Nếu mắt nhắm liên tục
            if state < 0.5:  # Mắt nhắm
                closed_frames += 1
            else:
                closed_frames = 0  # Đặt lại nếu mắt mở

    
    if closed_frames >= 5:
        cv2.putText(frame, "Ban dang ko tinh tao!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Cảnh Báo Buồn Ngủ", frame)

    # Hiển thị khung hình
    cv2.imshow('Drowsiness Detection', frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
