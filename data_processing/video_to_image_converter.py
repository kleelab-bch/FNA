"""
Author Junbong Jang
Date: 6/26/2019

"""
from subprocess import call
import os

def convert_video_automatic():
    path = "../assets/unstained"

    patient_folders = os.listdir(path)
    patient_folders = [path+"/"+patient_folder for patient_folder in patient_folders]
    # print(patient_folders) ['../assets/stained/2775', '../assets/stained/2776', '../assets/stained/2777', '../assets/stained/2779', '../assets/stained/2780', '../assets/stained/2781']

    for patient_folder_path in patient_folders:
        video_files = [f for f in os.listdir(patient_folder_path) if os.path.isfile(patient_folder_path + '/' + f)]
        video_file_names = [video_file.replace('.mp4', '') for video_file in video_files]
        video_file_paths = [patient_folder_path + "/" + video_file for video_file in video_files]
        for index, video_file_name in enumerate(video_file_names):
            image_folder_path = patient_folder_path + '/' + video_file_name
            print(image_folder_path)
            try:
                os.mkdir(image_folder_path)
                call(["ffmpeg", "-i", video_file_paths[index], "-r", "3",
                      image_folder_path + "/image-%04d.png"], shell=True)
                print('------------------------------------------------------\n\n\n\n\n\n')
            except OSError:
                print ("Creation of the directory %s failed" % path)
            else:
                print ("Successfully created the directory %s " % path)


if __name__ == "__main__":
    convert_video_automatic()
    # call(["ffmpeg", "-i", "../assets/stained/2776/2776-1A-3.mp4", "-r", "3", "../assets/stained/2776/2776-1A-3/image-%04d.png"], shell=True)