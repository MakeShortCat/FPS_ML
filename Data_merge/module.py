
# def process_col(df):
#     for cols in df.columns:
#         index_of_value = df[cols].eq(1).loc[lambda x: x].index - 1
#         index_of_value2 = df[cols].eq(2).loc[lambda x: x].index
#         inv1 = 0
#         inv2 = 0
#         while inv1 < len(index_of_value) and inv2 < len(index_of_value2):
#             if df.loc[index_of_value[inv1], cols] == 0:
#                 df.loc[index_of_value[inv1]+1:index_of_value2[inv2], cols] = 1
#                 inv1 += 1
#                 inv2 += 1
#             elif df.loc[index_of_value[inv1], cols] == 1:
#                 inv1 += 1
#                 inv2 = inv2
#     return df

def process_col(merged_log,cols):
    index_of_value = merged_log[cols].copy().eq(1).to_numpy().nonzero()[0] - 1
    index_of_value2 = merged_log[cols].copy().eq(2).to_numpy().nonzero()[0]
    inv1 = 0
    inv2 = 0
    while inv1 < len(index_of_value) and inv2 < len(index_of_value2):
        if merged_log[cols].iloc[index_of_value[inv1]] == 0:
            merged_log[cols].iloc[index_of_value[inv1]+1:index_of_value2[inv2]+1] = 1
            inv1 += 1
            inv2 += 1
        elif merged_log[cols].iloc[index_of_value[inv1]] == 1:
            inv1 += 1


def calculate_speed_for_target_num(target_num):
    speed_list = []
    target_num_log = merged_log[merged_log['target_num'] == target_num]
    for i in range(len(target_num_log) - 1):
        row = target_num_log.iloc[i]
        speed = calculate_speed(row['x'], row['y'], target_num_log.iloc[i+1]['x'], target_num_log.iloc[i+1]['y'], row['date_time'], target_num_log.iloc[i+1]['date_time'])
        speed_list.append(speed)
    speed_list.append(0)
    return speed_list