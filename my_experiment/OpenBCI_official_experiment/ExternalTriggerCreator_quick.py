# OpenBCI Experiment: ExternalTriggerCreator
# A python script to customize video components and external trigger for EEG experiments

# Date: 30/03/2022
# Author: Fan Li
# Editor: Lintao

import cv2
import glob
import argparse
import json
import random
import copy
import os
import numpy as np

random.seed(13)


def resize_image(old_size, new_size):
    ratio_y = new_size[0] / old_size[0]
    ratio_x = new_size[1] / old_size[1]

    if ratio_y > ratio_x:
        y = old_size[0] * ratio_x
        x = new_size[1]
    else:
        y = new_size[0]
        x = old_size[1] * ratio_y

    size = (int(y), int(x), new_size[2])
    return size


def embed_trigger(img, background_size, trigger_position, trigger_color):
    # background
    background = np.zeros(background_size, dtype="uint8")

    background[:] = (0, 0, 0)

    background[
    int(background_size[0] / 2) - int(img.shape[0] / 2):int(background_size[0] / 2) - int(img.shape[0] / 2) + img.shape[
        0], int(background_size[1] / 2) - int(img.shape[1] / 2): int(background_size[1] / 2) - int(img.shape[1] / 2) +
                                                                 img.shape[1]] = img

    # trigger
    background[trigger_position[1]:trigger_position[3], trigger_position[0]:trigger_position[2]] = (trigger_color)
    return background


###
def create_session(type_image_path_list, img_types, frame_numbers, fps, trigger_position):
    frames_array = []
    label_array = []
    label_index = []
    filename_array = []


    for i in range(1, len(type_image_path_list)+1):
        # multi class
        for filename in type_image_path_list[i-1]:

            # add current image
            img = cv2.imread(filename)
            frames, label = create_frames(img, frame_numbers, img_types[i-1], fps, trigger_position)
            frames_array.append(frames)
            label_array.append(label)
            label_index.append(i)
            filename_array.append(filename)


    # pack
    z = list(zip(frames_array, label_array, label_index, filename_array))
    random.shuffle(z)
    new_frame_array, new_label_array, new_label_index_array, new_filename_array = zip(*z)

    new_frame_array = list(new_frame_array)
    new_label_array = list(new_label_array)
    new_label_index_array = list(new_label_index_array)
    new_filename_array = list(new_filename_array)

    return new_frame_array, new_label_array, new_label_index_array, new_filename_array


def create_video(image_base_path, data_path, img_types, fps, flick_times, screen_size, time_range_per_image, video_output, label_output,
                 trigger_position):
    screen_size = tuple(screen_size)
    img_array = []

    data_path = image_base_path + data_path + "/"
    type_image_path_list = list()
    for img_type in img_types:
        type_image_path_list.append(glob.glob(data_path + img_type + '/*.jpg'))


    # session 1 1    s
    frame_numbers = int(fps * 1)
    # puppy_image_path = puppy_image_path_list[1]
    frame_array_session, label_array_session, label_index_array_session, filename_array_session = create_session(
        type_image_path_list, img_types, frame_numbers, fps, trigger_position)

    # # session 2 0.75 s
    # frame_numbers = int(fps * 0.75)
    # # puppy_image_path = puppy_image_path_list[2]
    # frame_array_session_2, label_array_session_2, label_index_array_session_2, filename_array_session_2 = create_session(
    #     type_image_path_list, img_types, frame_numbers, fps, trigger_position)

    # # session 3 0.5  s
    # frame_numbers = int(fps * 0.5)
    # # puppy_image_path = puppy_image_path_list[3]
    # frame_array_session_3, label_array_session_3, label_index_array_session_3, filename_array_session_3 = create_session(
    #     type_image_path_list, img_types, frame_numbers, fps, trigger_position)

    # # session 4 0.25 s
    # frame_numbers = int(fps * 0.25)
    # # puppy_image_path = puppy_image_path_list[4]
    # frame_array_session_4, label_array_session_4, label_index_array_session_4, filename_array_session_4 = create_session(
    #     type_image_path_list, img_types, frame_numbers, fps, trigger_position)

    new_frame_array = frame_array_session
    new_label_array = label_array_session
    new_label_index_array = label_index_array_session
    new_filename_array = filename_array_session

    for frames in new_frame_array:
        for image in frames:
            img_array.append(image)
            # print("output shape: ", image[0].shape)

    with open(label_output, 'w') as handle:
        for i in range(len(new_label_array)):
            handle.write("%s," % new_label_index_array[i])
            handle.write("%s," % new_label_array[i])
            local_file_name = new_filename_array[i].split("\\")[-1]
            handle.write("%s," % local_file_name)
            handle.write("\n")

    # Welcome frames
    welcome_array = []

    img = cv2.imread(image_base_path + 'Welcome/welcome2OpenBCI.jpg')

    ######
    new_size = (1080, 1350, 3)
    background_size = (1080, 1920, 3)
    old_size = img.shape
    size = resize_image(old_size, new_size)

    img_new = cv2.resize(img, (size[1], size[0]))

    img_white = embed_trigger(img_new, background_size, trigger_position, (255, 255, 255))
    img_black = embed_trigger(img_new, background_size, trigger_position, (0, 0, 0))

    for i in range(3 * fps):
        welcome_array.append(img_black)

    img_1 = cv2.imread(image_base_path + 'Welcome/1.PNG')
    old_size = img_1.shape
    size = resize_image(old_size, new_size)
    img_new_1 = cv2.resize(img_1, (size[1], size[0]))
    img_1_black = embed_trigger(img_new_1, background_size, trigger_position, (0, 0, 0))
    for i in range(3 * fps):
        welcome_array.append(img_1_black)

    img_2 = cv2.imread(image_base_path + 'Welcome/2.PNG')
    old_size = img_2.shape
    size = resize_image(old_size, new_size)
    img_new_2 = cv2.resize(img_2, (size[1], size[0]))
    img_2_black = embed_trigger(img_new_2, background_size, trigger_position, (0, 0, 0))
    for i in range(3 * fps):
        welcome_array.append(img_2_black)

    img_2 = cv2.imread(image_base_path + 'Welcome/2.PNG')

    for j in range(flick_times):
        for i in range(int(0.05 * fps)):
            welcome_array.append(img_black)
        for i in range(int(0.05 * fps)):
            welcome_array.append(img_white)

    # ending frames
    ending_array = []
    for j in range(flick_times):
        for i in range(int(0.05 * fps)):
            ending_array.append(img_black)
        for i in range(int(0.05 * fps)):
            ending_array.append(img_white)

    img_array = welcome_array + img_array + ending_array
    print(len(welcome_array), len(img_array))

    out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, screen_size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def create_frames(img, frame_num, label, fps, trigger_position):
    new_size = (1080, 1350, 3)
    background_size = (1080, 1920, 3)
    old_size = img.shape
    adjusted_size = resize_image(old_size, new_size)

    img_new = cv2.resize(img, (adjusted_size[1], adjusted_size[0]))

    print("Output Shape: ", img_new.shape)

    img_white = embed_trigger(img_new, background_size, trigger_position, (255, 255, 255))
    img_black = embed_trigger(img_new, background_size, trigger_position, (0, 0, 0))
    cv2.imwrite("test.jpg", img_white)

    # generate fixation across
    cross = np.zeros((1080, 1920, 3), np.uint8)
    cv2.line(cross, (950, 540), (970, 540), (0, 0, 255), 2)
    cv2.line(cross, (960, 530), (960, 550), (0, 0, 255), 2)

    # embed

    frame_array = []
    # random interval
    random_interval = random.randint(-2, 2) / 10
    # add cross before current image
    frame_array += [cross] * int((0.5 + random_interval) * fps)

    # add current image
    for i in range(int(0.05 * fps)):
        frame_array.append(img_white)
    for i in range(frame_num):
        frame_array.append(img_black)
    for i in range(int(0.05 * fps)):
        frame_array.append(img_black)

    # random interval
    # random_interval = random.randint(-2, 2) / 10

    # add cross after current image
    frame_array += [cross] * int((0.5 + random_interval) * fps)
    print(int((0.5 + random_interval) * fps))

    return frame_array, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load_json", help="load json to parse args")
    args = parser.parse_args()
    if args.load_json:
        with open(args.load_json, 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)
    create_video(args.image_base_path,
                 args.data_path,
                 args.image_type,
                 args.fps,
                 args.flick_times,
                 args.screen_size,
                 args.time_range_per_image,
                 args.video_output,
                 args.label_output,
                 args.trigger_position)
