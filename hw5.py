__author__ = "Chongye Wang, Si Chen, Pan Zhang"

import os
import numpy as np
from sklearn.cluster import KMeans
import sys

def divide(action_dict, length):
    new_action_dict = {}
    for action in action_dict:
        new_action_dict[action] = []
        for list in action_dict[action]:
            row_size = len(list) # The number of rows
            num_of_blocks = int(row_size / length)
            if num_of_blocks > 0: # Make sure there is at least one block
                for idx in range(0, num_of_blocks):
                    new_list = []
                    start_index = idx * length
                    end_index = (idx + 1) * length
                    for l in range(start_index, end_index):
                        new_list += list[l]
                    new_action_dict[action].append(new_list)
    return new_action_dict


def reverse_dict(action_dict):
    result = {}
    for key in action_dict:
        list = action_dict[key]
        for l in list:
            result[tuple(l)] = key
    return result

############################## Data preprocessing ##############################

# Get the current working directory
curr_directory = os.getcwd()

all_data = curr_directory + '/HMP_Dataset'

all_folders = [d for d in os.listdir(all_data) if os.path.isdir(os.path.join(all_data, d))]

# Get only folders without 'MODEL'
selected_folders = [d for d in all_folders if 'MODEL' not in d]

action_dict = {}

for folder in selected_folders:
    action_dict[folder] = []
    one_folder = all_data + '/' + folder
    all_files = os.listdir(one_folder)# All files in one folder
    for file in all_files: # One file in a folder
        file_path = one_folder + '/' + file
        with open(file_path, 'r') as f:
            lines = f.readlines()
        lines = [x.strip() for x in lines]
        for idx in range(len(lines)):
            lines[idx] = lines[idx].split(' ')
            lines[idx] = [int(x) for x in lines[idx]]
        action_dict[folder].append(lines)


################################################################################

#we do vector quantization
action_dict = divide(action_dict, 32)

# reverse the key and value of action_dict
label_action = reverse_dict(action_dict)


#Sum all the list
sum_all_list = []
for action in action_dict:
    for list in action_dict[action]:
        sum_all_list.append(list)
sum_all_list = np.array(sum_all_list)


#Cluster with KMeans(we use 14 clusters here)
kmeans = KMeans(n_clusters=14, random_state=0).fit(sum_all_list)
predict_result = kmeans.labels_


# Count the number of real actions
action_count = {}
for action in action_dict:
    action_count[action] = 0
for action in action_dict:
    action_count[action] = len(action_dict[action])
print(action_count)

predicted_count = {}
for idx in range(14):
    predicted_count[idx] = 0
for prediction in predict_result:
    predicted_count[prediction] += 1
print(predicted_count)

"""
{'Liedown_bed': 345, 'Walk': 2835, 'Eat_soup': 207, 'Getup_bed': 1381, 'Descend_stairs': 460,
'Use_telephone': 470, 'Standup_chair': 745, 'Brush_teeth': 926, 'Climb_stairs': 1210,
'Sitdown_chair': 727, 'Eat_meat': 974, 'Comb_hair': 722, 'Drink_glass': 1288, 'Pour_water': 1254}

{0: 907, 1: 2587, 2: 384, 3: 1747, 4: 724, 5: 543, 6: 486, 7: 1068, 8: 489,
9: 625, 10: 609, 11: 736, 12: 2488, 13: 151}
"""

action_cluster = {}

for prediction in predicted_count:
    prediction_count = predicted_count[prediction]
    corresponding_result = ''
    curr_min = sys.maxsize
    for action in action_count:
        real_count = action_count[action]
        difference = abs(prediction_count - real_count)
        if difference < curr_min:
            curr_min = difference
            corresponding_result = action
    action_cluster[prediction] = corresponding_result
print(action_cluster)
