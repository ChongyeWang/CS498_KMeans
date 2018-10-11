__author__ = "Chongye Wang, Si Chen, Pan Zhang"


import os
import numpy as np
from sklearn.cluster import KMeans
import sys
import numpy.linalg as la
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt



def divide(action_dict, length):
    """
    Divide the whole data sets.
    """
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
    """
    Reverse the key and value of a dictionary
    """
    result = {}
    for key in action_dict:
        list = action_dict[key]
        for l in list:
            result[tuple(l)] = key
    return result


def divide_one_file(list, length):
    """
    Divid one file.
    """
    row_size = len(list) # The number of rows
    num_of_blocks = int(row_size / length)
    result = []
    if num_of_blocks > 0: # Make sure there is at least one block
        for idx in range(0, num_of_blocks):
            new_list = []
            start_index = idx * length
            end_index = (idx + 1) * length
            for l in range(start_index, end_index):
                new_list += list[l]
            result.append(new_list)
    return result


def kmeans(k, n):

    """
    This method implements the k means clustering
    k : number of clusters
    n : number of dividing length
    """

    ############################## Data preprocessing ##############################
    ################################################################################

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
    ################################################################################

    #Represent each label with numbers
    action_num = {
        'Liedown_bed' : 0,
        'Walk' : 1,
        'Eat_soup' : 2,
        'Getup_bed' : 3,
        'Descend_stairs' : 4,
        'Use_telephone' : 5,
        'Standup_chair' : 6,
        'Brush_teeth' : 7,
        'Climb_stairs' : 8,
        'Sitdown_chair' : 9,
        'Eat_meat' : 10,
        'Comb_hair' : 11,
        'Drink_glass' : 12,
        'Pour_water' : 13
    }

    original = dict(action_dict)

    test = {}
    for action in action_dict:
        test[action] = []
    test_label = []


    for action in original:
        size = len(original[action])
        train_size = int(size * 0.75)
        for i in range(train_size, size):
            test[action].append(original[action][i])
            test_label.append(action_num[action])
        original[action] = original[action][0:train_size]


    #we do vector quantization
    action_dict = divide(original, n)

    # reverse the key and value of action_dict
    label_action = reverse_dict(action_dict)

    #Sum all the list
    sum_all_list = []
    for action in action_dict:
        for l in action_dict[action]:
            sum_all_list.append(l)
    sum_all_list = np.array(sum_all_list)


    #Cluster with KMeans(we use 14 clusters here)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(sum_all_list)
    cluster_center = kmeans.cluster_centers_

    #represent each center with number 0 - 13
    center_mark = {}
    num = 0
    for center in cluster_center:
        center_mark[tuple(center)] = num
        num +=1


    vec_label = {}

    for action in label_action:
        curr_action = list(action)
        min_distance = sys.maxsize
        label = 0
        for center in center_mark:
            curr_center = list(center)
            distance = la.norm(np.array(curr_center) - np.array(curr_action))
            if distance < min_distance:
                min_distance = distance
                label = center_mark[center]
        vec_label[action] = label


    action_center = {}
    for action in action_dict:
        action_center[action] = [0 for i in range(k)]
    for action in action_dict:
        all_action_vec = action_dict[action]
        for vec in all_action_vec:
            label = vec_label[tuple(vec)]
            action_center[action][label] += 1

    ################ Train ###############

    train = []
    train_label = []
    for action in original:
        all_files = original[action]
        for one_list in all_files:
            div = divide_one_file(one_list, n)
            result = [0 for i in range(k)]
            for curr_action in div:
                label = 0
                min_distance = sys.maxsize
                for center in center_mark:
                    curr_center = list(center)
                    distance = la.norm(np.array(curr_center) - np.array(curr_action))
                    if distance < min_distance:
                        min_distance = distance
                        label = center_mark[center]
                result[label] += 1
            train.append(result)
            train_label.append(action_num[action])

    #Process test data
    test_data = []
    for action in test:
        all_files = test[action]
        for one_list in all_files:
            div = divide_one_file(one_list, n)
            result = [0 for i in range(k)]
            for curr_action in div:
                label = 0
                min_distance = sys.maxsize
                for center in center_mark:
                    curr_center = list(center)
                    distance = la.norm(np.array(curr_center) - np.array(curr_action))
                    if distance < min_distance:
                        min_distance = distance
                        label = center_mark[center]
                result[label] += 1
            test_data.append(result)

    """
    #Support Vector Machine
    #Fit the model
    clf = svm.SVC(gamma='scale')
    clf.fit(train, train_label)
    #Get the result
    print(clf.score(test_data, test_label))
    """

    clf = RandomForestClassifier(n_estimators=100, max_depth=32)
    clf.fit(train, train_label)
    #Get the result
    print(clf.score(test_data, test_label))


    ##### confusion_matrix #####
    y_predict = clf.predict(test_data)
    y_true = test_label
    cm = confusion_matrix(y_true, y_predict)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("confusion_matrix")
    plt.colorbar()
    plt.show()
    ############################



    ################################################################################
    ################################################################################
    ###### This part is the overall pattern and will not be used for training ######

    """
    action_center:

    {'Liedown_bed': [18, 40, 17, 37, 15, 59, 20, 28, 2, 42, 0, 13, 54, 0],
    'Walk': [33, 1294, 0, 22, 6, 4, 150, 27, 1, 86, 24, 5, 1183, 0],
    'Eat_soup': [22, 0, 0, 21, 125, 0, 0, 39, 0, 0, 0, 0, 0, 0],
    'Getup_bed': [72, 322, 53, 122, 22, 144, 53, 81, 19, 194, 8, 121, 155, 15],
    'Descend_stairs': [12, 114, 0, 4, 0, 0, 48, 21, 0, 36, 1, 0, 224, 0],
    'Use_telephone': [35, 0, 17, 48, 9, 6, 1, 17, 241, 3, 3, 90, 0, 0],
    'Standup_chair': [59, 151, 0, 135, 5, 19, 28, 43, 0, 93, 7, 4, 201, 0],
    'Brush_teeth': [44, 2, 0, 2, 113, 11, 51, 9, 0, 19, 526, 0, 13, 136],
    'Climb_stairs': [24, 541, 0, 13, 12, 1, 66, 25, 5, 41, 7, 4, 471, 0],
    'Sitdown_chair': [116, 100, 0, 156, 20, 21, 43, 52, 0, 49, 16, 5, 149, 0],
    'Eat_meat': [18, 0, 124, 605, 4, 0, 0, 190, 0, 0, 0, 33, 0, 0],
    'Comb_hair': [37, 23, 127, 26, 84, 5, 22, 84, 173, 23, 3, 77, 38, 0],
    'Drink_glass': [249, 0, 34, 225, 111, 1, 4, 218, 48, 16, 5, 377, 0, 0],
    'Pour_water': [168, 0, 12, 331, 198, 272, 0, 234, 0, 23, 9, 7, 0, 0]}

    """

    """
    scaled_action_center = {}
    for action in action_dict:
        scaled_action_center[action] = [0 for i in range(14)]
    for ac in action_center:
        scaled_action_center[ac] = preprocessing.scale(action_center[ac])


    num_correct = 0
    num_incorrect = 0
    ## Train
    for action in original:
        all_lists = original[action]
        for one_list in all_lists:
            div = divide_one_file(one_list, 32)
            result = [0 for i in range(14)]
            for curr_action in div:
                label = 0
                min_distance = sys.maxsize
                for center in center_mark:
                    curr_center = list(center)
                    distance = la.norm(np.array(curr_center) - np.array(curr_action))
                    if distance < min_distance:
                        min_distance = distance
                        label = center_mark[center]
                result[label] += 1
            scaled_result = preprocessing.scale(result)

            predict = ''
            min_distance = sys.maxsize
            for sac in scaled_action_center:
                curr = scaled_action_center[sac]
                distance = la.norm(np.array(curr) - np.array(scaled_result))
                if distance < min_distance:
                    min_distance = distance
                    predict = sac
            if predict == action:
                num_correct += 1
            else:
                num_incorrect += 1
    print(num_correct / (num_correct + num_incorrect))

    """

    ################################################################################
    ################################################################################
if __name__ == "__main__":

    """
    for k in range(5, 30):
        kmeans(k, 32)

    for n in range(10, 30):
        kmeans(19, n)
    """
    kmeans(19, 12)
