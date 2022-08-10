import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import sys
from scipy.fftpack import fft, ifft

sampling_rate = 125


def read_data(path_list, need_all=False, need_XYZ=False):
    # data_alphabet_1 = './data/Tests_EEG_Lintao/alphabets_test_1.txt'
    # # data_alphabet_1 = os.path.join(datapath, 'alphabets_test_1.txt')
    # # data_alphabet_1 = os.path.join(os.path.dirname(os.getcwd()), 'data/Tests_EEG_Lintao/alphabets_test_1.txt')
    # data_alphabet_2 = './data/Tests_EEG_Lintao/alphabets_test_2.txt'
    # data_alphabet_3 = './data/Tests_EEG_Lintao/alphabets_test_3.txt'
    # data_alphabet_4 = './data/Tests_EEG_Lintao/alphabets_test_4.txt'
    # data_alphabet_5 = './data/Tests_EEG_Lintao/alphabets_test_5.txt'
    # data_alphabet_6 = './data/Tests_EEG_Lintao/alphabets_test_6.txt'
    # data_alphabet_X = './data/Tests_EEG_Lintao/alphabets_test_X.txt'
    #
    # data_asl_1 = './data/Tests_EEG_Lintao/asl_test_1.txt'
    # data_asl_2 = './data/Tests_EEG_Lintao/asl_test_2.txt'
    # data_asl_3 = './data/Tests_EEG_Lintao/asl_test_3.txt'
    # data_asl_4 = './data/Tests_EEG_Lintao/asl_test_4.txt'
    # data_asl_5 = './data/Tests_EEG_Lintao/asl_test_5.txt'
    # data_asl_6 = './data/Tests_EEG_Lintao/asl_test_6.txt'
    # data_asl_X = './data/Tests_EEG_Lintao/asl_test_X.txt'
    #
    # label_alphabet_1 = './data/Tests_EEG_Lintao/labels_alphabets_1.txt'
    # label_alphabet_2 = './data/Tests_EEG_Lintao/labels_alphabets_2.txt'
    # label_alphabet_3 = './data/Tests_EEG_Lintao/labels_alphabets_3.txt'
    # label_alphabet_4 = './data/Tests_EEG_Lintao/labels_alphabets_4.txt'
    # label_alphabet_5 = './data/Tests_EEG_Lintao/labels_alphabets_5.txt'
    # label_alphabet_6 = './data/Tests_EEG_Lintao/labels_alphabets_6.txt'
    # label_alphabet_X = './data/Tests_EEG_Lintao/labels_alphabets_X.txt'

    names = ['Index', 'ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'ch9', 'ch10', 'ch11', 'ch12', 'ch13',
             'ch14', 'ch15', 'ch16', 'accel_x', 'accel_y', 'accel_z', 'other1', 'other2', 'other3', 'other4', 'other5',
             'other6', 'other7', 'analog_ch1', 'analog_ch2', 'analog_ch3', 'TimeStamp', 'other8', 'time']

    alphabet1 = pd.read_csv(path_list[0], sep=",", header=6, index_col=False, names=names)
    alphabet2 = pd.read_csv(path_list[1], sep=",", header=6, index_col=False, names=names)
    alphabet3 = pd.read_csv(path_list[2], sep=",", header=6, index_col=False, names=names)
    alphabet4 = pd.read_csv(path_list[3], sep=",", header=6, index_col=False, names=names)
    alphabet5 = pd.read_csv(path_list[4], sep=",", header=6, index_col=False, names=names)
    alphabet6 = pd.read_csv(path_list[5], sep=",", header=6, index_col=False, names=names)
    alphabetX = pd.read_csv(path_list[6], sep=",", header=6, index_col=False, names=names)

    asl1 = pd.read_csv(path_list[7], sep=",", header=6, index_col=False, names=names)
    asl2 = pd.read_csv(path_list[8], sep=",", header=6, index_col=False, names=names)
    asl3 = pd.read_csv(path_list[9], sep=",", header=6, index_col=False, names=names)
    asl4 = pd.read_csv(path_list[10], sep=",", header=6, index_col=False, names=names)
    asl5 = pd.read_csv(path_list[11], sep=",", header=6, index_col=False, names=names)
    asl6 = pd.read_csv(path_list[12], sep=",", header=6, index_col=False, names=names)
    aslX = pd.read_csv(path_list[13], sep=",", header=6, index_col=False, names=names)

    labels_1 = pd.read_csv(path_list[14], sep=',', index_col=False, names=['label_index', 'label', 'filename'])
    labels_2 = pd.read_csv(path_list[15], sep=',', index_col=False, names=['label_index', 'label', 'filename'])
    labels_3 = pd.read_csv(path_list[15], sep=',', index_col=False, names=['label_index', 'label', 'filename'])
    labels_4 = pd.read_csv(path_list[17], sep=',', index_col=False, names=['label_index', 'label', 'filename'])
    labels_5 = pd.read_csv(path_list[18], sep=',', index_col=False, names=['label_index', 'label', 'filename'])
    labels_6 = pd.read_csv(path_list[19], sep=',', index_col=False, names=['label_index', 'label', 'filename'])
    labels_X = pd.read_csv(path_list[20], sep=',', index_col=False, names=['label_index', 'label', 'filename'])

    if need_all:
        return [alphabet1, alphabet2, alphabet3, alphabet4, alphabet5, alphabet6, alphabetX], \
                [asl1, asl2, asl3, asl4, asl5, asl6, aslX], \
                [labels_1, labels_2, labels_3, labels_4, labels_5, labels_6, labels_X]

    else:

        if need_XYZ:
            usecols = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'ch9', 'ch10',
                       'ch11', 'ch12', 'ch13', 'ch14', 'ch15', 'ch16', 'accel_x', 'accel_y', 'accel_z', 'TimeStamp',
                       'time']
            # choose the columns useful
            alphabet1_selected = alphabet1[usecols]
            alphabet2_selected = alphabet2[usecols]
            alphabet3_selected = alphabet3[usecols]
            alphabet4_selected = alphabet4[usecols]
            alphabet5_selected = alphabet5[usecols]
            alphabet6_selected = alphabet6[usecols]
            alphabetX_selected = alphabetX[usecols]

            asl1_selected = asl1[usecols]
            asl2_selected = asl2[usecols]
            asl3_selected = asl3[usecols]
            asl4_selected = asl4[usecols]
            asl5_selected = asl5[usecols]
            asl6_selected = asl6[usecols]
            aslX_selected = aslX[usecols]

            return [alphabet1_selected, alphabet2_selected, alphabet3_selected, alphabet4_selected, alphabet5_selected, alphabet6_selected, alphabetX_selected], \
                   [asl1_selected, asl2_selected, asl3_selected, asl4_selected, asl5_selected, asl6_selected, aslX_selected], \
                   [labels_1, labels_2, labels_3, labels_4, labels_5, labels_6, labels_X]

        else:
            usecols = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'ch9', 'ch10',
                       'ch11', 'ch12', 'ch13', 'ch14', 'ch15', 'ch16', 'TimeStamp', 'time']
            # choose the columns useful
            alphabet1_selected = alphabet1[usecols]
            alphabet2_selected = alphabet2[usecols]
            alphabet3_selected = alphabet3[usecols]
            alphabet4_selected = alphabet4[usecols]
            alphabet5_selected = alphabet5[usecols]
            alphabet6_selected = alphabet6[usecols]
            alphabetX_selected = alphabetX[usecols]

            asl1_selected = asl1[usecols]
            asl2_selected = asl2[usecols]
            asl3_selected = asl3[usecols]
            asl4_selected = asl4[usecols]
            asl5_selected = asl5[usecols]
            asl6_selected = asl6[usecols]
            aslX_selected = aslX[usecols]

            return [alphabet1_selected, alphabet2_selected, alphabet3_selected, alphabet4_selected, alphabet5_selected, alphabet6_selected, alphabetX_selected], \
                   [asl1_selected, asl2_selected, asl3_selected, asl4_selected, asl5_selected, asl6_selected, aslX_selected], \
                   [labels_1, labels_2, labels_3, labels_4, labels_5, labels_6, labels_X]


def get_intervals_data(df, named_X=False):
    # get timestamp array
    Timestamps = np.array(df['TimeStamp'] - df['TimeStamp'][0])

    # find the starting index of our range
    for i in range(len(Timestamps)):
        if Timestamps[i] - Timestamps[0] >= 18:
            starting = i
            break

    if named_X:
        # find the ending index of our range for X
        for i in range(len(Timestamps)):
            if Timestamps[i] - Timestamps[0] >= 554.8:
                ending = i - 1
                break
    else:
        # find the ending index of our range for number
        for i in range(len(Timestamps)):
            if Timestamps[i] - Timestamps[0] >= 774.4:
                ending = i - 1
                break

    # get the intervals
    indexs = list()
    length_vision = list()
    length_imagination = list()
    current = Timestamps[starting]
    current_index = starting
    vision = True
    current_missing = Timestamps[starting] - 18
    count_vision = 0
    count_imagination = 0

    for i in range(starting, ending + 1):

        if vision:
            if Timestamps[i] - current >= 3.1 - current_missing:
                count_vision += 1
                current_missing = Timestamps[i] - 18 - 3.1 * count_vision - 3.0 * count_imagination
                vision = False
                current = Timestamps[i]
                # use i-1 not i, make sure data interval is pure data
                indexs.append((current_index, i - 1))
                length_vision.append(i - 1 - current_index)
                current_index = i

        else:
            if Timestamps[i] - current >= 3.0 - current_missing:
                count_imagination += 1
                current_missing = Timestamps[i] - 18 - 3.1 * count_vision - 3.0 * count_imagination
                vision = True
                current = Timestamps[i]
                indexs.append((current_index, i - 1))
                length_imagination.append(i - 1 - current_index)
                current_index = i

    indexs.append((current_index, ending))

    if named_X:
        assert len(indexs) == 88 * 2
    else:
        assert len(indexs) == 124 * 2

    return indexs, length_vision, length_imagination


def pack_data(path_list, keep_same_lengths=False):

    alphabet_list, asl_list, label_list = read_data(path_list)

    indexs_alphabet1, length_vision1, length_imagination1 = get_intervals_data(alphabet_list[0])
    indexs_alphabet2, length_vision2, length_imagination2 = get_intervals_data(alphabet_list[1])
    indexs_alphabet3, length_vision3, length_imagination3 = get_intervals_data(alphabet_list[2])
    indexs_alphabet4, length_vision4, length_imagination4 = get_intervals_data(alphabet_list[3])
    indexs_alphabet5, length_vision5, length_imagination5 = get_intervals_data(alphabet_list[4])
    indexs_alphabet6, length_vision6, length_imagination6 = get_intervals_data(alphabet_list[5])
    indexs_alphabetX, length_visionX, length_imaginationX = get_intervals_data(alphabet_list[6], named_X=True)

    indexs_asl1, length_vision11, length_imagination11 = get_intervals_data(asl_list[0])
    indexs_asl2, length_vision22, length_imagination22 = get_intervals_data(asl_list[1])
    indexs_asl3, length_vision33, length_imagination33 = get_intervals_data(asl_list[2])
    indexs_asl4, length_vision44, length_imagination44 = get_intervals_data(asl_list[3])
    indexs_asl5, length_vision55, length_imagination55 = get_intervals_data(asl_list[4])
    indexs_asl6, length_vision66, length_imagination66 = get_intervals_data(asl_list[5])
    indexs_aslX, length_visionXX, length_imaginationXX = get_intervals_data(asl_list[6], named_X=True)


    alphabet_vision = list()
    alphabet_imagination = list()
    asl_vision = list()
    asl_imagination = list()

    # alphabet vision
    data_intervals = [indexs_alphabet1, indexs_alphabet2, indexs_alphabet3, indexs_alphabet4,
                      indexs_alphabet5, indexs_alphabet6, indexs_alphabetX]
    data_source_index = [0, 1, 2, 3, 4, 5, 6]

    for i in range(0, 7):
        label_index = label_list[i]["label_index"].tolist()
        label_name = label_list[i]["label"].tolist()

        for j in range(0, len(data_intervals[i]), 2):
            # print(label_index[j/2])
            current_data = (data_source_index[i], data_intervals[i][j], "vision", "alphabet", label_index[int(j / 2)],
                            label_name[int(j / 2)])
            alphabet_vision.append(current_data)

    # alphabet imagination
    for i in range(0, 7):
        label_index = label_list[i]["label_index"].tolist()
        label_name = label_list[i]["label"].tolist()

        for j in range(1, len(data_intervals[i]) + 1, 2):
            # print(label_index[j/2])
            current_data = (data_source_index[i], data_intervals[i][j], "imagination", "alphabet",
                            label_index[int(j / 2)], label_name[int(j / 2)])
            alphabet_imagination.append(current_data)

    # asl vision
    data_intervals = [indexs_asl1, indexs_asl2, indexs_asl3, indexs_asl4, indexs_asl5, indexs_asl6, indexs_aslX]
    for i in range(0, 7):
        label_index = label_list[i]["label_index"].tolist()
        label_name = label_list[i]["label"].tolist()

        for j in range(0, len(data_intervals[i]), 2):
            # print(label_index[j/2])
            current_data = (data_source_index[i], data_intervals[i][j], "vision", "asl",
                            label_index[int(j / 2)], label_name[int(j / 2)])
            asl_vision.append(current_data)

    # asl imagination
    for i in range(0, 7):
        label_index = label_list[i]["label_index"].tolist()
        label_name = label_list[i]["label"].tolist()

        for j in range(1, len(data_intervals[i]) + 1, 2):
            # print(label_index[j/2])
            current_data = (data_source_index[i], data_intervals[i][j], "imagination", "asl", label_index[int(j / 2)], label_name[int(j / 2)])
            asl_imagination.append(current_data)

    return alphabet_list, asl_list, alphabet_vision, alphabet_imagination, asl_vision, asl_imagination


def read_orginal_data(path_list):
    alphabet_list, asl_list, label_list = read_data(path_list)

    feature_path = "./data/EEG_features_Lintao/"
    alphabet_vision = pd.read_csv(feature_path+"aat_vision.csv", header=0,index_col=0)
    alphabet_imagination = pd.read_csv(feature_path+"aat_img.csv", header=0,index_col=0)
    asl_vision = pd.read_csv(feature_path+"asl_vision.csv", header=0,index_col=0)
    asl_imagination = pd.read_csv(feature_path+"asl_img.csv", header=0,index_col=0)

    return alphabet_list, asl_list, alphabet_vision, alphabet_imagination, asl_vision, asl_imagination


def read_power_band_txt():
    feature_path = "./data/EEG_features_Lintao/"
    bp_feature_path = feature_path + "band_power/"

    # Read all three methods power band
    filename_list = os.listdir(bp_feature_path)
    path_list = list()
    bp_data_dict = dict()
    strr_list = list()

    for item in filename_list:
        path_list.append(os.path.join(bp_feature_path, item))
        strr = item.split('_bp')[0]
        strr_list.append(strr)

    for i in range(len(path_list)):
        with open(path_list[i], 'r') as current:
            raw = current.readlines()
            bp = list()
            for item in raw:
                cc = list()
                item = item.split("\t")[2:-1]
                for j in item:
                    cc.append(eval(j))
                bp.append(cc)

            bp_data_dict[strr_list[i]] = bp

    return bp_data_dict


def read_seg_power_band_txt():
    feature_path = "./data/EEG_features_Lintao/"
    bp_feature_path = feature_path + "seg_band_powers/"

    # Read all three methods power band
    filename_list = os.listdir(bp_feature_path)
    path_list = list()
    bp_data_dict = dict()
    strr_list = list()

    for item in filename_list:
        path_list.append(os.path.join(bp_feature_path, item))
        strr = item.split('_bp')[0]
        strr_list.append(strr)

    for i in range(len(path_list)):
        with open(path_list[i], 'r') as current:
            raw = current.readlines()
            bp = list()
            for item in raw:
                cc = list()
                item = item.split("\t")[1:-1]
                for j in item:
                    cc.append(eval(j))
                bp.append(cc)

            bp_data_dict[strr_list[i]] = bp

    return bp_data_dict


def read_features_table():

    feature_path = "./data/EEG_features_Lintao/eeg_features/"
    alphabet_vision = pd.read_csv(feature_path+"aat_vision.csv", header=0, index_col=0)
    alphabet_imagination = pd.read_csv(feature_path+"aat_img.csv", header=0, index_col=0)
    asl_vision = pd.read_csv(feature_path+"asl_vision.csv", header=0, index_col=0)
    asl_imagination = pd.read_csv(feature_path+"asl_img.csv", header=0, index_col=0)

    return alphabet_vision, alphabet_imagination, asl_vision, asl_imagination


def read_seg_features_table():

    feature_path = "./data/EEG_features_Lintao/seg_eeg_features/"
    alphabet_vision = pd.read_csv(feature_path+"aat_vision.csv", header=0, index_col=0)
    alphabet_imagination = pd.read_csv(feature_path+"aat_img.csv", header=0, index_col=0)
    asl_vision = pd.read_csv(feature_path+"asl_vision.csv", header=0, index_col=0)
    asl_imagination = pd.read_csv(feature_path+"asl_img.csv", header=0, index_col=0)

    return alphabet_vision, alphabet_imagination, asl_vision, asl_imagination


def read_signal_data():

    feature_path = "./data/EEG_features_Lintao/eeg_signals/"
    alphabet_vision = pd.read_csv(feature_path+"aat_vision.csv", header=0, index_col=0)
    alphabet_imagination = pd.read_csv(feature_path+"aat_img.csv", header=0, index_col=0)
    asl_vision = pd.read_csv(feature_path+"asl_vision.csv", header=0, index_col=0)
    asl_imagination = pd.read_csv(feature_path+"asl_img.csv", header=0, index_col=0)

    return alphabet_vision, alphabet_imagination, asl_vision, asl_imagination


def read_seg_signal_data():
    feature_path = "./data/EEG_features_Lintao/seg_eeg_signals/"
    alphabet_vision = pd.read_csv(feature_path+"aat_vision.csv", header=0, index_col=0)
    alphabet_imagination = pd.read_csv(feature_path+"aat_img.csv", header=0, index_col=0)
    asl_vision = pd.read_csv(feature_path+"asl_vision.csv", header=0, index_col=0)
    asl_imagination = pd.read_csv(feature_path+"asl_img.csv", header=0, index_col=0)

    return alphabet_vision, alphabet_imagination, asl_vision, asl_imagination


def str_2_list(strr):
    strr = strr.replace("\n","").replace("  ", ",").replace(" ", ",").replace(",,",',').replace("[","").replace("]","")
    results = [float(n) for n in strr.split(',')]
    return results


if __name__ == '__main__':
    print(os.path.join(os.path.dirname(os.getcwd()), 'data/Tests_EEG_Lintao/alphabets_test_1.txt'))
    # make sure you understand the relative position path for this file or notebook file
    # alphabet_list, asl_list, alphabet_vision, alphabet_imagination, asl_vision, asl_imagination = pack_data()