from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import cv2
import numpy as np
import os

# Шлях до папки з обличчям
faces_folder = "faces"
# Шлях до папки без обличчя
non_faces_folder = "non-faces"


# Функція для завантаження та підготовки зображень
def load_and_prepare_data(folder, label):
    data = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if img_path.endswith(".jpg"):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))  # Змініть розмір зображення за потребою
            data.append(img)
            labels.append(label)
    return data, labels


# Завантаження та підготовка зображень обличчя
faces_data, faces_labels = load_and_prepare_data(faces_folder, 1)

# Завантаження та підготовка зображень без обличчя
non_faces_data, non_faces_labels = load_and_prepare_data(non_faces_folder, 0)

# Об'єднання даних та міток
X_train = np.array(faces_data + non_faces_data)
y_train = np.array(faces_labels + non_faces_labels)


# Побудова моделі
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Компіляція моделі
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Навчання моделі
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Збереження моделі
model.save('face_recognition_model.h5')
