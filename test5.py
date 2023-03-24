import cv2
import numpy as np

def check_color_range(log_file_path):
    cap = cv2.VideoCapture("C:/Users/pgs66/Desktop/test.mp4")
    # 프레임당 처리할 프레임 수
    frame_interval = 61

    # 특정 두 가지 색이 함께 존재하는 경우 건너뛸 프레임 수
    skip_frames = 306

    # 결과를 저장할 로그 파일 열기
    with open(log_file_path, 'w') as log_file:
        # 프레임 처리를 위한 루프 시작
        while cap.isOpened():
            # 현재 프레임 가져오기
            ret, frame = cap.read()

            # 프레임이 없으면 루프 종료
            if not ret:
                break

            for i in range(frame_interval - 1):
                cap.read()

            result = process_frame(frame, skip_frames, cap)

            # 결과를 로그 파일에 저장
            log_file.write(str(result) + '\n')

    # 동영상 파일 닫기
    cap.release()


kill_log_color = []
for i in range(210, 256):
    for j in range(210, 256):
        for k in range(90, 155):
            kill_log_color.append((i,j,k))

kill_log_color_red = []
for i in range(170, 256):
    for j in range(70, 141):
        for k in range(70, 141):
            kill_log_color_red.append((i,j,k))

def process_frame(frame, skip_frames, cap):
    left = 1300
    top = 0
    right = 1920
    bottom = 700
    color_values1 = set(tuple(frame[y][x]) for x in range(left, right) for y in range(top, bottom))

    leftd = 1800
    topd = 300
    rightd = 1920
    bottomd = 500
    color_values2 = set(tuple(frame[y][x]) for x in range(leftd, rightd) for y in range(topd, bottomd))

    # 리스트 -> 넘파이 배열로 변환
    color_values1 = np.array(list(color_values1))
    color_values2 = np.array(list(color_values2))
    kill_log_color_np = np.array(kill_log_color)
    kill_log_color_np_red = np.array(kill_log_color_red)
    print(np.any(np.all(kill_log_color_np_red == color_values1[:, np.newaxis], axis=2)), np.any(np.all(kill_log_color_np == color_values2[:, np.newaxis], axis=2)))

    # 각각의 픽셀 색상이 특정색상과 같은지 체크
    if np.any(np.all(kill_log_color_np == color_values1[:, np.newaxis], axis=2)):
        result = 'kill'
        for i in range(skip_frames - 1):
            cap.read()
    elif np.any(np.all(kill_log_color_np == color_values2[:, np.newaxis], axis=2)) and np.any(np.all(kill_log_color_np_red == color_values1[:, np.newaxis], axis=2)):
        result = 'death'
        for i in range(skip_frames - 1):
            cap.read()
    else:
        result = 0

    return result


check_color_range('FPS_ML_project/colorlog_cpu.txt')