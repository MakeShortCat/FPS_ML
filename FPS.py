import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# 이미지가 저장된 폴더 경로를 지정합니다.
image_dir = "C:/Users/pgs66/Desktop/FPSPICT"

# 이미지 전처리를 위한 ImageDataGenerator 객체를 생성합니다.
datagen = ImageDataGenerator(rescale=1./255)

# 이미지 데이터를 불러옵니다.
image_data = datagen.flow_from_directory(
    image_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

# 모델 구성을 위한 Sequential 객체를 생성합니다.
model = Sequential()

# Convolution layer와 MaxPooling layer를 추가합니다.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flatten layer를 추가합니다.
model.add(Flatten())

# Fully connected layer를 추가합니다.
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))

# 모델을 컴파일합니다.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델을 학습합니다.
model.fit(image_data, epochs=10)

# 학습된 모델을 저장합니다.
model.save('C:/Users/pgs66/Desktop/GoogleDrive/python/FPS_ML_project/model.h5')

import cv2
import numpy as np
import tensorflow as tf

# 모델 로드
model = tf.keras.models.load_model('FPS_ML_project/model.h5')

# 동영상 열기
cap = cv2.VideoCapture("E:/GameReplay/2023-03-04 18-31-50.mkv")

# 로그 파일 열기
log_file = open("FPS_ML_project/killlog.txt", 'w')

while True:
    # 프레임 읽기
    ret, frame = cap.read()

    # 프레임이 없으면 종료
    if not ret:
        break

    # 이미지 전처리
    image = cv2.resize(frame, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)

    # 예측
    predictions = model.predict(image)

    # 예측 결과를 로그 파일에 저장
    if np.any(predictions == 1):
        log_file.write('1\n')
    else:
        log_file.write('0\n')

    # 이미지 출력 (테스트용)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료
cap.release()
log_file.close()
cv2.destroyAllWindows()

