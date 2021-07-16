import numpy as np
import os, cv2
import random
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split

class prediction_data_generate:
    def __init__(self, img_path, random_seed, img_format = '.png', rand_crop_num = 500):
        self.img_path = img_path
        self.random_seed = random_seed
        self.img_format = img_format
        self.rand_crop_num = rand_crop_num
        self.row, self.col = self.get_img_size()

    #==================================================================================
    #==================================================================================
    def get_whole_frames(self):
        # get the namespace
        namespace = self.find_namespace()
       
        imgs, image_rows, image_cols = self.testing_read(self.img_path, namespace)
        return imgs, namespace, image_cols, image_rows, self.col, self.row
    
    #==================================================================================
    #==================================================================================
    def find_namespace(self):
        namespace = []
        img_path = self.img_path
        
        img_file_name = os.listdir(img_path)
        for file in img_file_name:
            if os.path.isfile(img_path + file) and file.endswith(self.img_format):
                namespace.append(file)
        #print(namespace)
        return namespace

    #==================================================================================
    #==================================================================================
    def get_img_size(self): # for training set
        
        img_path = self.img_path
        namespace = self.find_namespace()
        
        for file in namespace:
            #Find the mapping filename in the mask folder
            filemask = file
            if os.path.isfile(img_path + file) and file.endswith(self.img_format):
                return cv2.imread(img_path + file , cv2.IMREAD_GRAYSCALE).shape
        print("invalid imgs")
        return -1, -1

    #==================================================================================
    #==================================================================================
    def testing_read(self,img_path, namelist, ratio = 64.0): # read images within namelist
        total_number = len(namelist)
        imgs_row_exp = int(np.ceil(np.divide(self.row, ratio) ) * ratio)
        imgs_col_exp = int(np.ceil(np.divide(self.col, ratio) ) * ratio)
        
        #applying space to save the results
        imgs = np.ndarray((total_number, int(imgs_row_exp), int(imgs_col_exp)), dtype=np.uint8) 
        i = 0
        for name in namelist:
            cv2.imread(img_path + name, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize( cv2.imread(img_path + name, cv2.IMREAD_GRAYSCALE) ,(int(self.col), int(self.row)), interpolation = cv2.INTER_CUBIC)
            imgs[i] = cv2.copyMakeBorder(img, 0, imgs_row_exp - self.row, 0, imgs_col_exp - self.col, cv2.BORDER_REFLECT)
            i += 1
        return imgs, imgs_row_exp, imgs_col_exp