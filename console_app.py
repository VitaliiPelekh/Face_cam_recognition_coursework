import cv2
import dlib
import numpy as np
from keras.models import load_model

# Завантаження моделі для розпізнавання обличчя
model = load_model('face_recognition_model.h5')  # Призначте шлях до вашої моделі

# Ініціалізація детектора обличчя dlib
face_detector = dlib.get_frontal_face_detector()

# Ініціалізація спеціального дескриптора обличчя dlib для визначення ключових точок
shape_predictor = dlib.shape_predictor(
    'shape_predictor_68_face_landmarks.dat')  # Призначте шлях до файлу з ключовими точками


# Функція для розпізнавання обличчя на вхідному зображенні
def recognize_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    for face in faces:
        landmarks = shape_predictor(gray, face)
        face_image = gray[face.top():face.bottom(), face.left():face.right()]

        # Перевірка, чи обличчя знайдено і його розмір не порожній
        if not face_image.size:
            continue

        # Зміна розміру зображення обличчя
        face_image = cv2.resize(face_image, (64, 64))
        face_image = np.expand_dims(face_image, axis=0) / 255.0
        prediction = model.predict(face_image)[0]

        if prediction[0] > 0.45:
            cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
            cv2.putText(image, 'Face Detected', (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

    return image



# Відкриття вебкамери
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = recognize_face(frame)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
