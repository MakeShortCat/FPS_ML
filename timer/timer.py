from tkinter import *
import datetime

# Tkinter 창 생성
root = Tk()

# 창 크기 설정
root.geometry("633x70")

# Label 위젯 생성
label = Label(root, font=('calibri', 40, 'bold'), background='black', foreground='white')
label.pack(anchor='center')

# 화면에 실시간으로 업데이트하는 함수
def update_label():
    current_time = datetime.datetime.now()
    label.config(text=current_time)
    root.after(25, update_label)

# update_label 함수 호출
update_label()

# Tkinter 창 실행
root.mainloop()
