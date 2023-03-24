import cv2

def check_color_range(video_path):
    # 동영상 파일 열기
    cap = cv2.VideoCapture(video_path)

    # 프레임당 처리할 프레임 수
    frame_interval = 1

    # 프레임 처리를 위한 루프 시작
    while cap.isOpened():
        # 현재 프레임 가져오기
        ret, frame = cap.read()

        # 프레임이 없으면 루프 종료
        if not ret:
            break

        # 특정 범위의 색상값 가져오기
        left = 0
        top = 0
        right = 100
        bottom = 100
        color_values = set(frame[y][x] for x in range(left, right) for y in range(top, bottom))

        # 특정 두 가지 색이 함께 존재하는지 확인
        if (0, 0, 255) in color_values and (255, 0, 0) in color_values:
            return 1

    # 동영상 파일 닫기
    cap.release()

    # 특정 두 가지 색이 함께 존재하지 않는 경우 0 반환
    return 0

import tensorflow as tf

def train_mnist():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize the data to have values between 0 and 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

    # Save the model
    model.save('mnist_model.h5')

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the MNIST model
model = load_model('mnist_model.h5')

# Define a function to preprocess the image
def preprocess_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to 28x28
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    # Invert the image
    inverted = cv2.bitwise_not(resized)

    # Normalize the pixel values to be between 0 and 1
    normalized = inverted / 255.0

    # Reshape the image to have a single channel
    reshaped = np.reshape(normalized, (1, 28, 28, 1))

    return reshaped

# Load the video file
cap = cv2.VideoCapture('number_video.mp4')

# Open the output text file
with open('numbers.txt', 'w') as f:
    # Loop through the frames of the video
    while True:
        # Read the next frame
        ret, frame = cap.read()

        # If the frame is not read correctly, break out of the loop
        if not ret:
            break

        # Preprocess the frame
        processed = preprocess_image(frame)

        # Make a prediction with the model
        prediction = model.predict(processed)

        # Write the predicted number to the output file
        f.write(str(np.argmax(prediction)))

# Release the video capture and close the output file
cap.release()
f.close()