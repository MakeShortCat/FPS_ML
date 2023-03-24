import cv2
import cupy as cp

def check_color_range(log_file_path):
    cap = cv2.VideoCapture("C:/Users/pgs66/Desktop/test2.mp4")
    # 프레임당 처리할 프레임 수
    frame_interval = 10

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
for i in range(120, 130):
    for j in range(230, 240):
        for k in range(230, 240):
            kill_log_color.append((i,j,k))

kill_log_color_black = [(255,255,255), (255,255,254)]

def process_frame(frame, skip_frames, cap):
    left = 1381
    top = 120
    right = 1600
    bottom = 265
    color_values1 = set(tuple(frame[y][x]) for x in range(left, right) for y in range(top, bottom))

    left2 = 1400
    top2 = 1360
    right2 = 1450
    bottom2 = 1380
    color_values3 = set(tuple(frame[y][x]) for x in range(left2, right2) for y in range(top2, bottom2))


    # CPU -> GPU
    color_values1 = cp.array(list(color_values1))
    color_values3 = cp.array(list(color_values3))
    kill_log_color_gpu = cp.array(kill_log_color)
    kill_log_color_cp_black = cp.array(kill_log_color_black)

    # 각각의 픽셀 색상이 특정색상과 같은지 체크
    if cp.any(cp.all(kill_log_color_gpu == color_values1[:, cp.newaxis], axis=2)):
        result = 'kill'
        for i in range(skip_frames - 1):
            cap.read()
    elif not cp.any(cp.all(kill_log_color_cp_black == color_values3[:, cp.newaxis], axis=2)):
        result = 'death'
        for i in range(skip_frames - 1):
            cap.read()
    else:
        result = 0

    return result


check_color_range('FPS_ML_project/colorlog_gpu_d.txt')


