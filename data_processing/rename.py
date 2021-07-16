import numpy as np
import os


def operate_on_image_list():
    path = "../assets/unstained"

    patient_folders = os.listdir(path)
    patient_folders = [path + "/" + patient_folder for patient_folder in patient_folders]
    print(patient_folders)
    # print(patient_folders) ['../assets/stained/2775', '../assets/stained/2776', '../assets/stained/2777', '../assets/stained/2779', '../assets/stained/2780', '../assets/stained/2781']

    for patient_folder_path in patient_folders:  # iterate through the patient folders
        video_files = [f for f in os.listdir(patient_folder_path) if os.path.isfile(patient_folder_path + '/' + f)]
        print(f'patient folder: {patient_folder_path}')
        video_file_names = [video_file.replace('.mp4', '') for video_file in video_files]
        for index, video_file_name in enumerate(video_file_names): # iterate within a patient folder
            image_folder_path = patient_folder_path + '/' + video_file_name
            print('+++++++++++++++++++++ New Image Path List ++++++++++++++++++++')
            print(f'image folder: {image_folder_path}')
            image_path_list = [image_folder_path + '/' + f for f in os.listdir(image_folder_path) if
                               os.path.isfile(image_folder_path + '/' + f)]
            iterate_images(video_file_name, image_path_list)
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n\n\n')


def iterate_images(video_file_name, image_path_list):
    for index, image_path in enumerate(image_path_list):
        video_file_name
        dst = image_path.replace('image', video_file_name)
        # rename() function will rename all the files
        os.rename(image_path, dst)


def rename_labeled():
    patient_folder_path = "./mask_all"
    image_files = [f for f in os.listdir(patient_folder_path) if os.path.isfile(patient_folder_path + '/' + f)]
    print('---------------------')
    print(image_files)
    stored_images = []
    for image_file in image_files:
        if "L.png" in image_file:
            print('hello')
            dst = image_file.replace('L.png', '.png')
            # rename() function will rename all the files
            os.rename(patient_folder_path + '/' + image_file, dst)


if __name__ == "__main__":
    operate_on_image_list()
