import os
import shutil
import numpy as np

NUM_PER_CLASS = 2
np.random.seed(36) # 77, 13, 67, 99, 7, 36
data_path2 = "./Images/ASL_letters/"
data_path1 = "./Images/Alphabet_letters_upper/"
new_path1 = "./Images/Letter_class/"
new_path2 = "./Images/Sign_class/"

if not os.path.exists(new_path1):
    os.makedirs(new_path1)
if not os.path.exists(new_path2):
    os.makedirs(new_path2)

def choose_img(img_names, nums):
    new_list = list()
    for i in nums:
        new_list.append(img_names[i])   
    return new_list


files1 = os.listdir(data_path1)
files2 = os.listdir(data_path2)
# print(files1, files2)

for file in files1:
    # only open rep file
    file_path = os.path.join(data_path1, file)
    if os.path.isdir(file_path):   
        current_files = os.listdir(file_path)

        # random choose some images, then copy in another rep and change their names  
        
        images_chosed = np.random.randint(0, 50, NUM_PER_CLASS)
        current_files_chosed = choose_img(current_files, images_chosed)
        # current_files_chosed = lambda(x: current_files[x], images_chosed)

        for i in range(NUM_PER_CLASS):
            img_path = os.path.join(file_path, current_files_chosed[i])           
            if not os.path.exists(new_path1 + file + "/"):
                os.makedirs(new_path1 + file + "/")
            new_img_path = new_path1 + file + "/" + file + "_" + str(i+1) + ".png"
            shutil.copyfile(img_path, new_img_path)
            # os.rename(current_files_chosed)

for file in files2:
    # only open rep file
    file_path = os.path.join(data_path2, file)
    if os.path.isdir(file_path):
        current_files = os.listdir(file_path)

        # random choose some images, then copy in another rep and change their names  
        
        images_chosed = np.random.randint(0, len(current_files), NUM_PER_CLASS)
        current_files_chosed = choose_img(current_files, images_chosed)

        for i in range(NUM_PER_CLASS):
            img_path = os.path.join(file_path, current_files_chosed[i])           
            if not os.path.exists(new_path2 + file + "/"):
                os.makedirs(new_path2 + file + "/")
            new_img_path = new_path2 + file + "/" + file + "_" + str(i+1) + ".png"
            shutil.copyfile(img_path, new_img_path)
