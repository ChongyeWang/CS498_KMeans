__author__ = "Chongye Wang, Si Chen, Pan Zhang"

import os

def divid(action_dict, k):
    new_action_dict = {}
    for action in action_dict:
        new_action_dict[action] = []
        for list in action_dict[action]:
            row_size = len(list) # The number of rows
            num_of_blocks = int(row_size / (k * 3))
            if num_of_blocks > 0: # Make sure there is at least one block of size k * 3
                for idx in range(0, num_of_blocks):
                    new_list = [] # size of (1, k * 3)
                    start_index = idx * k
                    end_index = (idx + 1) * k
                    for l in range(start_index, end_index):
                        new_list += list[l]

                    new_action_dict[action].append(new_list)
                    print(new_action_dict)
    print(new_action_dict)
    return new_action_dict




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

divid(action_dict, 32)
