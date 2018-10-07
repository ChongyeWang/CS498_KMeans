__author__ = "Chongye Wang, Si Chen, Pan Zhang"

import os
import numpy as np
from sklearn.cluster import KMeans


def divid(action_dict, length):
    new_action_dict = {}
    for action in action_dict:
        new_action_dict[action] = []
        for list in action_dict[action]:
            row_size = len(list) # The number of rows
            num_of_blocks = int(row_size / length)
            if num_of_blocks > 0: # Make sure there is at least one block of size length * 3
                for idx in range(0, num_of_blocks):
                    new_list = [] # size of (1, length * 3)
                    start_index = idx * length
                    end_index = (idx + 1) * length
                    for l in range(start_index, end_index):
                        new_list += list[l]

                    new_action_dict[action].append(new_list)
                    print(new_action_dict)
    print(new_action_dict)
    return new_action_dict

def reverse_dict(action_dict):
    result = {}
    for key in action_dict:
        list = action_dict[key]
        for l in list:
            result[tuple(l)] = key
    return result



# Get the current working directory
curr_directory = os.getcwd()

all_data = curr_directory + '/HMP_Dataset'

all_folders = [d for d in os.listdir(all_data) if os.path.isdir(os.path.join(all_data, d))]

# Get only folders without 'MODEL'
selected_folders = [d for d in all_folders if 'MODEL' not in d]


data = []

action_dict = {}


for folder in selected_folders:
    print(folder)
    action_dict[folder] = []
    one_folder = all_data + '/' + folder
    all_files = os.listdir(one_folder)# All files in one folder
    for file in all_files: # One file in a folder
        file_path = one_folder + '/' + file
        with open(file_path, 'r') as f:
            lines = f.readlines()
        lines = [x.strip() for x in lines] #
        for idx in range(len(lines)):
            lines[idx] = lines[idx].split(' ')
            lines[idx] = [int(x) for x in lines[idx]]
        action_dict[folder].append(lines)


action_dict = divid(action_dict, 32)
label_action = reverse_dict(action_dict)

int_name = {}


sum_all_list = []
for action in action_dict:
    for list in action_dict[action]:
        sum_all_list.append(list)
sum_all_list = np.array(sum_all_list)


#Cluster with KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(sum_all_list)
predict_result = kmeans.labels_


#Test
num_correct = 0
for idx in range(len(sum_all_list)):
    preidct_result = predict_result[idx]
    estimated_result = predict_result[idx]
