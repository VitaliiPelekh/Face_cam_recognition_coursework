import cv2
import dlib
import curses
import threading
import numpy as np
from keras.models import load_model

# Глобальні змінні
sensitivity = 0.45  # Початкова чутливість розпізнавання
face_detection_enabled = True

# Завантаження моделі для розпізнавання обличчя
model = load_model('face_recognition_model.h5')  # Призначте шлях до вашої моделі

# Ініціалізація детектора обличчя dlib
face_detector = dlib.get_frontal_face_detector()

# Ініціалізація спеціального дескриптора обличчя dlib для визначення ключових точок
shape_predictor = dlib.shape_predictor(
    'shape_predictor_68_face_landmarks.dat')  # Призначте шлях до файлу з ключовими точками


# Оновлення параметрів розпізнавання
def update_recognition_parameters(new_sensitivity, enabled):
    global sensitivity, face_detection_enabled
    sensitivity = new_sensitivity
    face_detection_enabled = enabled


# Потік для розпізнавання облич
def face_recognition_thread():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if face_detection_enabled:
            frame = recognize_face(frame, sensitivity)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Функція розпізнавання обличчя з врахуванням чутливості
def recognize_face(image, sensitivity):
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

        if prediction[0] > sensitivity:
            cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
            cv2.putText(image, 'Face Detected', (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

    return image


def main(stdscr):
    global sensitivity, face_detection_enabled  # Вказуємо на глобальні змінні

    # Ініціалізація curses
    curses.curs_set(0)
    stdscr.nodelay(1)
    stdscr.clear()

    while True:
        stdscr.clear()
        stdscr.addstr("Face Recognition Console\n")
        stdscr.addstr(f"Sensitivity: {sensitivity:.3f}\n")
        stdscr.addstr("Press 'q' to quit, 'e' to enable/disable face recognition, '+' or '-' to adjust sensitivity\n")

        key = stdscr.getch()

        if key == ord('q'):
            break
        elif key == ord('e'):
            face_detection_enabled = not face_detection_enabled
        elif key == ord('+'):
            sensitivity = min(1.0, sensitivity + 0.005)
        elif key == ord('-'):
            sensitivity = max(0.0, sensitivity - 0.005)

        # Оновлення параметрів розпізнавання
        update_recognition_parameters(sensitivity, face_detection_enabled)

        stdscr.refresh()


# Запуск потоку для розпізнавання облич
threading.Thread(target=face_recognition_thread).start()

# Запуск curses application
curses.wrapper(main)
