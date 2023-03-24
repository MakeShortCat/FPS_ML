import cv2
import pytesseract
import cupy as cp
import datetime

date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H")
date_str_folder = datetime.datetime.now().strftime("%Y-%m-%d")
pytesseract.pytesseract.tesseract_cmd = r'C:/Users/pgs66/AppData/Local/tesseract.exe'

kill_counter = 0
death_counter = 0
no_kill_counter = 0

def check_color_range(log_file_path, video_path):
    cap = cv2.VideoCapture(video_path)
    # 프레임당 처리할 프레임 수
    frame_interval = 10

    # 킬로그가 발생시 건너뛸 프레임 수
    skip_frames = 306

    #몇번째 프레임인지 진행상황
    frame_nums = 0
    # 결과를 저장할 로그 파일 열기
    with open(log_file_path, 'w', encoding='UTF-8') as log_file:
        # 프레임 처리를 위한 루프 시작
        while cap.isOpened():
            # 현재 프레임 가져오기
            ret, frame = cap.read()
            print(f'{frame_nums}0번째 프레임 완료')
            frame_nums = frame_nums + 1
            # 프레임이 없으면 루프 종료
            if not ret:
                break

            for i in range(frame_interval - 1):
                cap.read()

            kill_result = process_frame(frame, skip_frames, cap)

            # 결과를 로그 파일에 저장
            if kill_result != None:
                log_file.write(str(kill_result) + '\n')

    # 동영상 파일 닫기
    cap.release()

# 색이 반투명한 재질이라 배경에 색상 값이 달라져서 상황별 색 값 범위 지정

#킬로그 노란계열
kill_log_color = []
for i in range(120, 133):
    for j in range(230, 240):
        for k in range(230, 240):
            kill_log_color.append((i,j,k))

#죽음뒤 검은색
kill_log_color_black = [(255,255,255), (255,255,254)]

#스킬 창 색
kill_log_color_skill = []
for i in range(95, 105):
    for j in range(120, 130):
        for k in range(50, 60):
            kill_log_color_skill.append((i,j,k))

#관전자 표시 색
kill_log_color_spect = []
for i in range(215, 245):
    for j in range(230, 251):
        for k in range(160, 186):
            kill_log_color_spect.append((i,j,k))

# 각 counter 들이 일정 이상 감지되야 로그가 찍히게 해서 스쳐지나가는 색 방지
# no_kill_counter 가 일정 이상 되면 카운터 초기화
def process_frame(frame, skip_frames, cap):
    global kill_counter
    global death_counter
    global no_kill_counter

#   킬로그 범위 (범위값은 프리미어 프로(영상편집 툴) 사용하여 알아냄)
    left = 1250
    top = 120
    right = 1580
    bottom = 265
    color_values1 = set(tuple(frame[y][x]) for x in range(left, right) for y in range(top, bottom))

    # left1 = 1080
    # top1 = 285
    # right1 = 1920
    # bottom1 = 770
    # color_values2 = set(tuple(frame[y][x]) for x in range(left1, right1) for y in range(top1, bottom1))

#   스킬 존재 범위
    left2 = 458
    top2 = 1315
    right2 = 1450
    bottom2 = 1430
    color_values3 = set(tuple(frame[y][x]) for x in range(left2, right2) for y in range(top2, bottom2))

#   관전자 버튼 범위
    left3 = 175
    top3 = 1035
    right3 = 190
    bottom3 = 1060
    color_values4 = set(tuple(frame[y][x]) for x in range(left3, right3) for y in range(top3, bottom3))


    # CPU -> GPU
    color_values1 = cp.array(list(color_values1))
    # color_values2 = cp.array(list(color_values2))
    color_values3 = cp.array(list(color_values3))
    color_values4 = cp.array(list(color_values4))
    kill_log_color_gpu = cp.array(kill_log_color)
    kill_log_color_cp_black = cp.array(kill_log_color_black)
    kill_log_color_cp_skill = cp.array(kill_log_color_skill)
    kill_log_color_cp_spect = cp.array(kill_log_color_spect)

    # 각각의 픽셀 색상이 특정색상과 같은지 체크
    if cp.any(cp.all(kill_log_color_gpu == color_values1[:, cp.newaxis], axis=2)) and not cp.any(cp.all(kill_log_color_cp_spect == color_values4[:, cp.newaxis], axis=2)):
        # 이미지에서 숫자 영역을 잘라내기
        img_gray = cv2.cvtColor(frame[0:60, 0:544], cv2.COLOR_BGR2GRAY)
        img_number = img_gray

        # OCR 수행하여 숫자 추출
        number = pytesseract.image_to_string(img_number, config='--psm 7')
        kill_counter +=1
        if kill_counter == 4:
            result = f'kill,{number}'
            kill_counter = 0
            for i in range(skip_frames - 1):
                cap.read()        
        else: result = None

    # 각각의 픽셀 색상이 특정색상과 같은지 체크
    elif not cp.any(cp.all(kill_log_color_cp_black == color_values3[:, cp.newaxis], axis=2))\
        and not cp.any(cp.all(kill_log_color_cp_skill == color_values3[:, cp.newaxis], axis=2)) \
            and not cp.any(cp.all(kill_log_color_cp_spect == color_values4[:, cp.newaxis], axis=2)):
        img_gray = cv2.cvtColor(frame[0:60, 0:540], cv2.COLOR_BGR2GRAY)
        img_number = img_gray

        # OCR 수행하여 숫자 추출
        number = pytesseract.image_to_string(img_number, config='--psm 6')
        death_counter += 1
        if death_counter == 9:
            result = f'death,{number}'
            death_counter = 0
            # 죽었을시 건너뛸 프레임(1분)
            for i in range(3600):
                cap.read()
        else: result = None

#   카운터 관리
    else:
        result = None
        no_kill_counter +=1
        if no_kill_counter == 60:
            kill_counter = 0
            death_counter = 0
            no_kill_counter = 0

    return result

check_color_range(f'FPS_ML_project/logs/2023-03-12/kill_2023-03-18 20.txt', "D:/GameReplay/2023-03-18 20-14-25.mkv")
