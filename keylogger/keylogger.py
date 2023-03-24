import os
import datetime
import time
from pynput import mouse, keyboard
import keyboard as kb

date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H")
date_str_folder = datetime.datetime.now().strftime("%Y-%m-%d")

# 키보드 로그 파일 경로
keyboard_file_path = f"C:/Users/pgs66/Desktop/GoogleDrive/python/FPS_ML_project/logs/{date_str_folder}/keyboard_{date_str}.txt"

# 마우스 로그 파일 경로
mouse_file_path = f"C:/Users/pgs66/Desktop/GoogleDrive/python/FPS_ML_project/logs/{date_str_folder}/keyboard_{date_str}.txt"

new_dir_path = f"C:/Users/pgs66/Desktop/GoogleDrive/python/FPS_ML_project/logs/{date_str_folder}/"

# 키보드 로그 파일이 있는지 확인하고 없으면 새 파일 생성
if not os.path.exists(keyboard_file_path):
    os.makedirs(new_dir_path, exist_ok=True)
    open(keyboard_file_path, "w").close()

# 마우스 로그 파일이 있는지 확인하고 없으면 새 파일 생성
if not os.path.exists(mouse_file_path):
    os.makedirs(new_dir_path, exist_ok=True)
    open(mouse_file_path, "w").close()

# 키보드 이벤트 핸들러
def on_press(key):
    try:
        current_time = datetime.datetime.now()
        with open(keyboard_file_path, "a") as f:
            f.write(f"{current_time}: {key} pressed\n")
            f.flush()  # 버퍼에 남아 있는 데이터 모두 출력
    except Exception as e:
        print(e)

def on_release(key):
    try:
        current_time = datetime.datetime.now()
        with open(keyboard_file_path, "a") as f:
            f.write(f"{current_time}: {key} released\n")
            f.flush()  # 버퍼에 남아 있는 데이터 모두 출력
    except Exception as e:
        print(e)
        
# 마우스 이벤트 핸들러
def on_move(x, y):
    try:
        current_time = datetime.datetime.now()
        with open(mouse_file_path, "a") as f:
            f.write(f"{current_time}: Mouse moved to ({x}, {y})\n")
            f.flush()  # 버퍼에 남아 있는 데이터 모두 출력
    except Exception as e:
        print(e)

# 마우스 클릭 이벤트 핸들러
def on_click(x, y, button, pressed):
    try:
        current_time = datetime.datetime.now()
        with open(mouse_file_path, "a") as f:
            f.write(f"{current_time}: {'Mouse' if pressed else 'Mouse released'} {button} at ({x}, {y})\n")
            f.flush()  # 버퍼에 남아 있는 데이터 모두 출력
    except Exception as e:
        print(e)

# 마우스 이벤트 리스너 객체 생성
mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click)

# 키보드 이벤트 리스너 객체 생성
keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)

# 키보드 이벤트 리스너 시작
keyboard_listener.start()

# 마우스 이벤트 리스너 시작
mouse_listener.start()

# ESC 키 입력 시 프로그램 종료
while True:
    if kb.is_pressed('ctrl+F9'):
        break
    time.sleep(0.1)

# 마우스 이벤트 리스너 중지
mouse_listener.stop()
keyboard_listener.stop()
