import pandas as pd
import numpy as np
from datetime import datetime, timedelta

date_str = datetime.now().strftime("%Y-%m-%d-%H")
date_str_folder = datetime.now().strftime("%Y-%m-%d")

# kill_log = pd.read_csv(f'FPS_ML_project/logs/{date_str_folder}/kill_{date_str}.txt')

kill_log = pd.read_csv('FPS_ML_project/logs/2023-03-20/kill_2023-03-20 00.txt',
                        header=None, names=['action', 'date_time'])

input_log = pd.read_table('FPS_ML_project/logs/2023-03-20/keyboard_2023-03-20-00.txt', sep=':\s',
                          header=None, names = ['date_time', 'action'], engine='python')

input_log = input_log[input_log['date_time'].str.startswith(f'{datetime.now().strftime("%Y")}')]

merged_log = pd.concat([kill_log, input_log], axis=0)

merged_log['date_time'] = pd.to_datetime(merged_log.date_time)

merged_log.sort_values('date_time', ascending=True, inplace=True)

merged_log.reset_index(drop=True, inplace=True)

merged_log['actual_action'] = np.nan

merged_log['target_num'] = np.nan

merged_log = merged_log[['date_time', 'action', 'actual_action', 'target_num']]

merged_log['actual_action'] = merged_log['action'].str.replace('Mouse moved to ', '')

merged_log

index1 = merged_log[merged_log['actual_action'].str.startswith('(') == True].index

merged_log['actual_action_1'] = np.nan

merged_log.loc[index1, 'actual_action_1'] = merged_log.loc[index1, 'actual_action']

merged_log['actual_action_1'] = merged_log['actual_action_1'].copy().str.replace('(', '',regex=True)
merged_log['actual_action_1'] = merged_log['actual_action_1'].copy().str.replace(')', '',regex=True)

merged_log[['x', 'y']] = merged_log['actual_action_1'].str.split(',', n=1, expand=True)

merged_log[['x', 'y']] = merged_log[['x', 'y']].apply(pd.to_numeric, errors='coerce').astype('Int64')

key_cols = ['w', 'a', 's', 'd', 'ctrl', 'q', 'e', 'c', 'shift']

merged_log[key_cols] = 0

for cols in key_cols:
    merged_log[cols] = merged_log['action'].apply(lambda x: 1 if x ==  f"'{cols}' pressed" else
                                                        2 if x== f"'{cols}' released"
                                                        else 0)

merged_log['click_left'] = merged_log['action'].apply(lambda x: 1 if x.startswith('Mouse Button.left') else
                                                    2 if x.startswith("Mouse released Button.left")
                                                    else 0)

merged_log['click_right'] = merged_log['action'].apply(lambda x: 1 if x.startswith('Mouse Button.right') else
                                                    2 if x.startswith("Mouse released Button.right")
                                                    else 0)

key_cols.extend(['click_left','click_right'])

import numpy as np
for col in key_cols:
    s = merged_log[col].values
    mask1 = np.where(s == 1)[0]
    mask2 = np.where(s == 2)[0]

    for i in range(len(mask1)):
        start = mask1[i]
        end = mask2[np.searchsorted(mask2, start, side='left')]
        s[start:end] = 1

    merged_log[col] = s

kill_num = 0

for i in range(len(merged_log)):
    if merged_log['action'].iloc[i] == 'kill':
        kill_num += 1
        for j in range(i - 1500, i):
            if merged_log['date_time'].iloc[j] > merged_log['date_time'].iloc[i] - timedelta(seconds=2.2) and \
               merged_log['date_time'].iloc[j] < merged_log['date_time'].iloc[i] - timedelta(seconds=0.2):
                merged_log['target_num'].iloc[j] = f'kill_{str(kill_num)}'

death_num = 0

for i in range(1501, len(merged_log)):
    if merged_log['action'].iloc[i] == 'death':
        death_num +=1
        for j in range(i - 1500, i):
            if merged_log['date_time'].iloc[j] > merged_log['date_time'].iloc[i] - timedelta(seconds=3.2) and \
               merged_log['date_time'].iloc[j] < merged_log['date_time'].iloc[i] - timedelta(seconds=1.2):
                merged_log['target_num'].iloc[j] = f'death_{str(death_num)}'

merged_log.dropna(subset=['target_num'], inplace=True)

merged_log.drop(['action', 'actual_action', 'actual_action_1'], axis=1, inplace=True)

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# 속도 계산 함수
def calculate_speed(x1, y1, x2, y2, t1, t2):
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    time_gap = (t2 - t1).total_seconds()
    # time_gap이 0이거나 t2가 NaN인 경우, 속도를 0으로 반환합니다.
    if time_gap == 0 or pd.isna(t2) or pd.isna(x2) or pd.isna(x1):
        speed = 0
    else:
        speed = distance / time_gap
    return speed

def calculate_speeds_for_target_num(target_num):
    target_num_log = merged_log[merged_log['target_num'] == target_num]
    speed_list = []
    for i in range(len(target_num_log) - 1):
        row = target_num_log.iloc[i]
        speed = calculate_speed(row['x'], row['y'], target_num_log.iloc[i+1]['x'], target_num_log.iloc[i+1]['y'], row['date_time'], target_num_log.iloc[i+1]['date_time'])
        speed_list.append(speed)
    return pd.Series(speed_list, index=target_num_log.index[:-1])

if __name__ == '__main__':
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(calculate_speeds_for_target_num, target_num) for target_num in merged_log.target_num.unique()]
        result = pd.concat([future.result() for future in as_completed(futures)])
    merged_log['speed'] = result

FPS_player_data = pd.DataFrame(columns=['target_num','Mouse_Acceleration_Mean', 'Mouse_Acceleration_Max',
                                        'Mouse_Total_Distance', 'Mouse_X_Distance', 'Mouse_Y_Distance',
                                        'Mouse_Pressed_Distance', 'Mouse-Keybord', 'Keyboard_pressed_Time',
                                        'Skill_Pressed'])

FPS_player_data['target_num'] = merged_log.groupby('target_num').aggregate(['mean', 'max'])['speed'].index

FPS_player_data['Mouse_Acceleration_Mean'] = merged_log.groupby('target_num').mean(numeric_only=True)['speed'].values

FPS_player_data['Mouse_Acceleration_Max'] = merged_log.groupby('target_num').max(numeric_only=True)['speed'].values

mouse_keybords_summery = pd.DataFrame(columns = ['target_num', 'Mouse-Keybord_mean', 'Mouse-Keybord_max', 'Mouse-Keybord_min'])

for target_num in merged_log.target_num.unique():
    mouse_keybords = []
    key_press = merged_log[merged_log['target_num'] == target_num][['w','a','s','d','click_left']].values
    key_press1 = merged_log[merged_log['target_num'] == target_num][['w','a','s','d']].values
    mouse_press = merged_log[merged_log['target_num'] == target_num]['click_left'].values
    mask1 = np.where(np.all(key_press) == 0)[0]
    mask2 = np.where(mouse_press == 1)[0]
    mask3 = np.where(np.any(key_press1) == 1)[0]
    diff1 = np.diff(mask1)
    indices1 = np.where(diff1 != 1)[0] + 1
    diff2 = np.diff(mask2)
    indices2 = np.where(diff2 != 1)[0] + 1
    indices2 = np.insert(indices2, 0, 0)
    diff3 = np.diff(mask3)
    indices3 = np.where(diff3 != 1)[0]
    indices3 = np.insert(indices3, 0, 0)


    if len(mask2) == 0:
        mouse_keybords_summery = mouse_keybords_summery.append({'target_num' : target_num,
                                                        'Mouse-Keybord_mean' : 0, 
                                                        'Mouse-Keybord_max' : 0, 
                                                        'Mouse-Keybord_min' : 0}, ignore_index = True)
        continue

    if len(indices1) > 0 and len(indices1) == len(indices2):
        for i, j in zip(indices1, indices2):
            mouse_keybords.append((merged_log[merged_log['target_num'] == target_num]['date_time'].iloc[j] - merged_log[merged_log['target_num'] == target_num]['date_time'].iloc[i]).total_seconds())

    else: mouse_keybords.append((merged_log[merged_log['target_num'] == target_num]['date_time'].iloc[mask2[0]] - merged_log[merged_log['target_num'] == target_num]['date_time'].iloc[mask1[-1]]).total_seconds())

    mouse_keybords_summery = mouse_keybords_summery.append({'target_num' : target_num,
                                                            'Mouse-Keybord_mean' : np.mean(mouse_keybords), 
                                                            'Mouse-Keybord_max' : np.max(mouse_keybords), 
                                                            'Mouse-Keybord_min' : np.min(mouse_keybords)}, ignore_index = True)
key_press_time = []
key_press_time_sum = []
for target_num in merged_log.target_num.unique():
    key_press1 = merged_log[merged_log['target_num'] == target_num][['w','a','s','d']].values
    mask3 = np.where(key_press1== 1)[0]
    diff3 = np.diff(mask3)
    indices3 = np.where(diff3 != 1)

    if len(mask3) == 0:
       key_press_time_sum.append(0)

       continue

    if len(indices3) > 1:
        for i in range(0, len(indices3)-1):
            key_press_time.append((merged_log[merged_log['target_num'] == target_num]['date_time'].iloc[indices3[i+1]] - merged_log[merged_log['target_num'] == target_num]['date_time'].iloc[indices3[i]]).total_seconds())

    else: key_press_time.append((merged_log[merged_log['target_num'] == target_num]['date_time'].iloc[mask3[-1]] - merged_log[merged_log['target_num'] == target_num]['date_time'].iloc[mask3[1]]).total_seconds())

    key_press_time_sum.append(np.sum(key_press_time))

    key_press_time = []
    


mouse_keybords_summery['key_press_time'] = key_press_time_sum

grouped = merged_log.groupby('target_num')

total_distance = grouped.apply(lambda x: np.sum(np.sqrt((x['x'] - x['x'].shift())**2 + (x['y'] - x['y'].shift())**2)))
x_distance = grouped.apply(lambda x: np.sum(np.abs(x['x'] - x['x'].shift())))
y_distance = grouped.apply(lambda x: np.sum(np.abs(x['y'] - x['y'].shift())))

FPS_player_data['Mouse_Total_Distance'] = total_distance.values

FPS_player_data['Mouse_X_Distance'] = x_distance.values

FPS_player_data['Mouse_Y_Distance'] = y_distance.values

FPS_player_data.drop(['Mouse_Pressed_Distance', 'Mouse-Keybord', 'Keyboard_pressed_Time', 'Skill_Pressed'], axis=1, inplace=True)

FPS_player_data = pd.merge(FPS_player_data, mouse_keybords_summery, on='target_num')

FPS_player_data.to_csv('C:/Users/pgs66/Desktop/GoogleDrive/python/FPS_ML_project/Data_merge/merged_data/2023-03-20 00.csv')
