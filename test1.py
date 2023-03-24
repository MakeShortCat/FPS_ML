import cv2
import numpy as np
def check_color_range(video_path, log_file_path):

    # 동영상 파일 열기
    cap = cv2.VideoCapture(video_path)

    # 프레임당 처리할 프레임 수
    frame_interval = 1

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

            # 특정 범위의 색상값 가져오기

            left = 1440 - 400
            top = 0
            right = 1340
            bottom = 600
            color_values1 = set(tuple(frame[y][x]) for x in range(left, right) for y in range(top, bottom))

            leftd = 1440 - 100
            topd = 0
            rightd = 1440
            bottomd = 600

            color_values2 = set(tuple(frame[y][x]) for x in range(leftd, rightd) for y in range(topd, bottomd))
            # 특정 색이 존재하는지 확인
            if (234, 233, 110) in color_values1:
                result = 'kill'

                # 특정 프레임 수만큼 건너뛰기
                for i in range(skip_frames - 1):
                    cap.read()
            else:
                result = 0

            if (234, 233, 110) in color_values2:
                result = 'death'

                # 특정 프레임 수만큼 건너뛰기
                for i in range(skip_frames - 1):
                    cap.read()
            else:
                result = 0
            

            # 결과를 로그 파일에 저장
            log_file.write(str(result) + '\n')

    # 동영상 파일 닫기
    cap.release()

check_color_range("E:/GameReplay/2023-03-04 18-31-50.mkv", 'FPS_ML_project/colorlog.txt')

