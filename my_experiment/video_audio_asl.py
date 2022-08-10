import cv2
import glob
import argparse
import json
import random
import copy
import os
import numpy as np
from audio_add import video_add_audio, video_concat

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


# get all frames by the alphabet_labels order
def create_session(type_image_path_list, img_types, frame_numbers, fps, trigger_position, label_position):
    frames_array = []
    label_array = []
    label_index = []
    filename_array = []

    # getting the labels from alphabets_labels
    with open(label_position, 'r') as handle:
        lines = handle.readlines()
        for item in lines:          
            current = item.split(",")
            label_index.append(current[0])
            label_array.append(current[1])
            filename_array.append(current[2])


    # creating frames in the alphabets labels order 
    for current_name in filename_array:
        for item in type_image_path_list:
            if current_name in item:
                break
            else:
                continue
        # item str has our current_name
        # print(item)
        img = cv2.imread(item)
        # cv2.imshow("a", img)
        # cv2.waitKey(4)
        frames = create_frames(img, frame_numbers, fps, trigger_position)
        frames_array.append(frames)
    
    # print(len(frames_array[1]))

    return frames_array, label_array, label_index, filename_array


def create_video(image_base_path, data_path, img_types, fps, flick_times, screen_size, time_range_per_image, video_output, label_position,
                 trigger_position):
    screen_size = tuple(screen_size)
    
    data_path = image_base_path + data_path + "/"
    type_image_path_list = list()
    for img_type in img_types:
        type_image_path_list.extend(glob.glob(data_path + img_type + '/*.png'))


    # 3 second
    frame_numbers = int(fps * 3)
    frame_array_session, label_array_session, label_index_array_session, filename_array_session = create_session(
        type_image_path_list, img_types, frame_numbers, fps, trigger_position, label_position)

    new_frame_array = frame_array_session


    # Welcome frames
    welcome_array = []

    img = cv2.imread(image_base_path + 'Welcome/ASL1.png')

    ######
    new_size = (1080, 1350, 3)
    background_size = (1080, 1920, 3)
    old_size = img.shape
    size = resize_image(old_size, new_size)

    img_new = cv2.resize(img, (size[1], size[0]))

    img_white = embed_trigger(img_new, background_size, trigger_position, (255, 255, 255))
    img_black = embed_trigger(img_new, background_size, trigger_position, (0, 0, 0))

    for i in range(5 * fps):
        welcome_array.append(img_black)

    img_1 = cv2.imread(image_base_path + 'Welcome/ASL2.png')

    old_size = img_1.shape
    size = resize_image(old_size, new_size)
    img_new_1 = cv2.resize(img_1, (size[1], size[0]))
    img_1_black = embed_trigger(img_new_1, background_size, trigger_position, (0, 0, 0))
    for i in range(5 * fps):
        welcome_array.append(img_1_black)

    img_2 = cv2.imread(image_base_path + 'Welcome/ASL3.png')
    old_size = img_2.shape
    size = resize_image(old_size, new_size)
    img_new_2 = cv2.resize(img_2, (size[1], size[0]))
    img_2_black = embed_trigger(img_new_2, background_size, trigger_position, (0, 0, 0))
    for i in range(5 * fps):
        welcome_array.append(img_2_black)

    img_2 = cv2.imread(image_base_path + 'Welcome/ASL3.png')

    for j in range(flick_times):
        for i in range(int(0.05 * fps)):
            welcome_array.append(img_black)
        for i in range(int(0.05 * fps)):
            welcome_array.append(img_white)

    # ending frames
    ending_array = []
    cross = np.zeros((1080, 1920, 3), np.uint8)
    cv2.line(cross, (950, 540), (970, 540), (0, 0, 255), 2)
    cv2.line(cross, (960, 530), (960, 550), (0, 0, 255), 2)
    ending_array += [cross] * int(2.5 * fps)
    for j in range(flick_times):
        for i in range(int(0.05 * fps)):
            ending_array.append(img_black)
        for i in range(int(0.05 * fps)):
            ending_array.append(img_white)

    # generate a whole video without audio and in order alphabets labels
    # img_array = []  
    # for frames in new_frame_array:
    #     for image in frames:
    #         img_array.append(image)
    # img_array = welcome_array + img_array + ending_array
    # out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, screen_size)
    # for i in range(len(img_array)):
    #     out.write(img_array[i])
    # out.release()
    
    # return

    # generate a whole video with audio and in order alphabets labels
    # start video
    start = cv2.VideoWriter("./asl_piece/start_noaudio.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, screen_size)
    for i in range(len(welcome_array)):
        start.write(welcome_array[i])
    start.release()

    # add A to start audio, if do not run this code, the following video won't have vocie
    video_add_audio("./asl_piece/start_noaudio.mp4", "./alphabet_sounds/add_silence/A.mp4", "./asl_piece/start.mp4")

    # generate video pieces
    for j in range(len(new_frame_array)):
        img_array = list()
        current = cv2.VideoWriter("./asl_piece/current.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, screen_size)
        for image in new_frame_array[j]:
            img_array.append(image)
        for i in range(len(img_array)):
            current.write(img_array[i])
        current.release()

        output_name = "./asl_piece/current_" + str(j) + ".mp4"
        # return 
        # add audio
        video_add_audio("./asl_piece/current.mp4", "./alphabet_sounds/add_silence/" + label_array_session[j] + ".mp4", output_name)
        

        # return 
        # concat video pieces
        if j == 0:
            output_name_2 = "./asl_piece/start_" + str(j) + ".mp4"
            video_concat("./asl_piece/start.mp4", output_name, output_name_2)

        # elif j == len(new_frame_array)-1:
        #     video_concat(output_name_2, output_name, video_output)
        else:
            t = output_name_2 
            output_name_2 = "./asl_piece/start_" + str(j) + ".mp4"
            video_concat(t, output_name, output_name_2)
    
    # concat ending video
    end = cv2.VideoWriter("./asl_piece/end.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, screen_size)
    for i in range(len(ending_array)):
        end.write(ending_array[i])
    end.release()

    video_concat(output_name_2, "./asl_piece/end.mp4", video_output)
    
            
def create_frames(img, frame_num, fps, trigger_position):
    new_size = (1080, 1350, 3)
    background_size = (1080, 1920, 3)
    old_size = img.shape
    adjusted_size = resize_image(old_size, new_size)

    img_new = cv2.resize(img, (adjusted_size[1], adjusted_size[0]))

    # print("Output Shape: ", img_new.shape)

    img_white = embed_trigger(img_new, background_size, trigger_position, (255, 255, 255))
    img_black = embed_trigger(img_new, background_size, trigger_position, (0, 0, 0))
    # cv2.imwrite("test.jpg", img_white)

    # generate fixation across
    cross = np.zeros((1080, 1920, 3), np.uint8)
    cv2.line(cross, (950, 540), (970, 540), (0, 0, 255), 2)
    cv2.line(cross, (960, 530), (960, 550), (0, 0, 255), 2)

    # embed

    frame_array = []

    # add cross before current image
    frame_array += [cross] * int(2.5 * fps)

    # add current image
    for i in range(int(0.05 * fps)):
        frame_array.append(img_white)
    for i in range(frame_num):
        frame_array.append(img_black)
    for i in range(int(0.05 * fps)):
        frame_array.append(img_black) 

    # add cross after current image
    frame_array += [cross] * int(0.5 * fps)

    return frame_array


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
                 "Sign_class",
                 args.image_types,
                 args.fps,
                 args.flick_times,
                 args.screen_size,
                 args.time_range_per_image,
                 "video_audio_asl_X.mp4",
                 "./labels_alphabets/labels_alphabets_X.txt",
                 args.trigger_position)
