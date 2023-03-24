import cv2
import pytesseract
import cupy as cp

pytesseract.pytesseract.tesseract_cmd = r'C:/Users/pgs66/AppData/Local/tesseract.exe'

kill_counter = 0
death_counter = 0
no_kill_counter = 10
def check_color_range(log_file_path):
    cap = cv2.VideoCapture("C:/Users/pgs66/Desktop/test3.mp4")
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

kill_log_color_skill = []
for i in range(99, 102):
    for j in range(54, 57):
        for k in range(99, 101):
            kill_log_color_skill.append((i,j,k))

kill_log_color_spect = []
for i in range(200, 210):
    for j in range(160, 170):
        for k in range(230, 240):
            kill_log_color_spect.append((i,j,k))


def process_frame(frame, skip_frames, cap):
    global kill_counter
    global death_counter
    global no_kill_counter

    left = 1381
    top = 120
    right = 1600
    bottom = 265
    color_values1 = set(tuple(frame[y][x]) for x in range(left, right) for y in range(top, bottom))

    left1 = 1080
    top1 = 285
    right1 = 1920
    bottom1 = 770
    color_values2 = set(tuple(frame[y][x]) for x in range(left1, right1) for y in range(top1, bottom1))

    left2 = 458
    top2 = 1315
    right2 = 1450
    bottom2 = 1430
    color_values3 = set(tuple(frame[y][x]) for x in range(left2, right2) for y in range(top2, bottom2))

    left3 = 175
    top3 = 1035
    right3 = 190
    bottom3 = 1060
    color_values4 = set(tuple(frame[y][x]) for x in range(left3, right3) for y in range(top3, bottom3))


    # CPU -> GPU
    color_values1 = cp.array(list(color_values1))
    color_values2 = cp.array(list(color_values2))
    color_values3 = cp.array(list(color_values3))
    color_values4 = cp.array(list(color_values4))
    kill_log_color_gpu = cp.array(kill_log_color)
    kill_log_color_cp_black = cp.array(kill_log_color_black)
    kill_log_color_cp_skill = cp.array(kill_log_color_skill)
    kill_log_color_cp_spect = cp.array(kill_log_color_spect)

    # 각각의 픽셀 색상이 특정색상과 같은지 체크
    if cp.any(cp.all(kill_log_color_gpu == color_values1[:, cp.newaxis], axis=2)) and not cp.any(cp.all(kill_log_color_cp_spect == color_values4[:, cp.newaxis], axis=2))\
          and not cp.any(cp.all(kill_log_color_gpu == color_values2[:, cp.newaxis], axis=2)):
        # 이미지에서 숫자 영역을 잘라내기
        img_gray = cv2.cvtColor(frame[0:62, 0:875], cv2.COLOR_BGR2GRAY)
        img_number = img_gray

        # OCR 수행하여 숫자 추출
        number = pytesseract.image_to_string(img_number, config='--psm 8')
        kill_counter +=1
        if kill_counter == 12:
            result = f'kill {number}'
            kill_counter = 0
            for i in range(skip_frames - 1):
                cap.read()        
        else: result = 0

    elif not cp.any(cp.all(kill_log_color_cp_black == color_values3[:, cp.newaxis], axis=2))\
        and not cp.any(cp.all(kill_log_color_cp_skill == color_values3[:, cp.newaxis], axis=2)):
        img_gray = cv2.cvtColor(frame[0:62, 0:875], cv2.COLOR_BGR2GRAY)
        img_number = img_gray

        # OCR 수행하여 숫자 추출
        number = pytesseract.image_to_string(img_number, config='--psm 8')
        death_counter += 1
        if death_counter == 6:
            result = f'death {number}'
            death_counter = 0
            for i in range(skip_frames - 1):
                cap.read()
        else: result = 0

    else:
        result = 0
        no_kill_counter +=1
        if no_kill_counter == 30:
            kill_counter = 0
            death_counter = 0

    return result

check_color_range('FPS_ML_project/colorlog_gpu_d.txt')