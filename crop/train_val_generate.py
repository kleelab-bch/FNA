import numpy as np
import os, cv2
import glob
#import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import backend as K
#K.set_image_dim_ordering('th')  # Theano dimension ordering in this code
K.set_image_data_format('channels_first')
smooth = 1.


def preprocess_output(imgs, img_rows, img_cols):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    imgs_p = imgs_p.astype('float32')
    imgs_p /= 255.  # scale masks to [0, 1]
    return imgs_p

def preprocess_input(imgs, img_rows, img_cols, mean, std):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
        imgs_p[i, 1] = cv2.resize(imgs[i, 1], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
        imgs_p[i, 2] = cv2.resize(imgs[i, 2], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
       
    imgs_p = imgs_p.astype('float32')
    imgs_p -= mean
    imgs_p /= std
    return imgs_p

def augment_data(imgs,msks,edgs,iteration):
    # define data preparation
    batch_size = 128
    datagen = ImageDataGenerator(
        rotation_range=50.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect')
    print('augment_data')
    print(imgs.shape)
    print(msks.shape)
    print(edgs.shape)
    #imgs = imgs[:,np.newaxis,:,:,:]
    #imgs = np.swapaxes(imgs,1,3)
    #imgs = np.swapaxes(imgs,2,3)
    #msks = msks[:,np.newaxis,:,:]
    edgs = edgs[:,np.newaxis,:,:]
        
    train = np.zeros((iteration*batch_size, imgs.shape[1], imgs.shape[2], imgs.shape[3])).astype('uint8')
    mask = np.zeros((iteration*batch_size, 1, msks.shape[2], msks.shape[3])).astype('uint8')
    edge = np.zeros((iteration*batch_size, 1, edgs.shape[2], edgs.shape[3])).astype('uint8')
    
    
    print('Data Generating...')
    for samples in range(iteration): 
        for imags_batch in datagen.flow(imgs, batch_size=batch_size, seed = samples): #probably can change "activation" parameter
            break 
        for mask_batch in datagen.flow(msks, batch_size=batch_size, seed = samples): 
            break
        for edge_batch in datagen.flow(edgs, batch_size=batch_size, seed = samples): 
            break
        train[samples*batch_size:(samples+1)*batch_size] = imags_batch
        mask[samples*batch_size:(samples+1)*batch_size] = mask_batch
        edge[samples*batch_size:(samples+1)*batch_size] = edge_batch
    print('Data Generation done')

    train = np.vstack([imgs, train])
    mask = np.vstack([msks, mask])
    edge = np.vstack([edgs, edge])
    
    mask = mask[:,:,30:98,30:98]
    edge = edge[:,:,30:98,30:98]
    
    train, mask, edge = shuffle(train, mask, edge, random_state=10)
    return train, mask, edge


class data_generate:
    def __init__(self, path, n_frames_train, input_size, output_size, random_seed, saved_folder, img_format = '.png', rand_crop_num = 200, root = '../../DataSet_label/', img_folder = '/img/', mask_folder = '/mask/'):
        self.n_frames_train = n_frames_train
        self.path = path
        self.random_seed = random_seed
        self.input_size = input_size
        self.output_size = output_size
        self.img_format = img_format
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.rand_crop_num = rand_crop_num
        self.saved_folder = saved_folder
        self.root = root
        self.row, self.col, self.total_frames = self.get_row_col()
        
        self.n_frames_val = 0
        self.n_frames_train = int(self.total_frames)
        #self.n_frames_val = int(self.total_frames * .8 * .2)
        #self.n_frames_train = int(self.total_frames * .8) - self.n_frames_val
        self.n_frames_test = self.total_frames - self.n_frames_train - self.n_frames_val
    #==================================================================================
    # Get the size of image and number of images
    #==================================================================================
    def get_row_col(self):
        path = self.root +  self.path + self.mask_folder
        mask_list = glob.glob(path + '*' + self.img_format)
        img = cv2.imread(mask_list[0], cv2.IMREAD_GRAYSCALE)
        r, c = img.shape
        return float(r), float(c), len(mask_list)
    #==================================================================================
    # Get the size of image and number of images
    #==================================================================================
    def read_msk(self, msk_f):
        msk = cv2.imread(msk_f, cv2.IMREAD_GRAYSCALE)       
        print(msk_f, np.unique(msk), end=' ')
        # for multi-label classification
        msk[msk==76] = 255 # follicular
        msk[msk==149] = 0 # secretion
        msk[msk==29] = 0 # artifact
        msk[(msk>0) & (msk!=127) & (msk!=255)] = 0 # for rest of the labels
        #msk[msk>0] = 255
        edg = cv2.Canny(msk,100,200)
        edg[edg>0] = 255
        print(np.unique(msk))
        
        return msk, edg
    #==================================================================================
    #==================================================================================
    def r_img_msk(self):
        r_path = self.root + self.path + self.img_folder
        m_path = self.root + self.path + self.mask_folder
        # mask_list = glob.glob(os.path.join(m_path, '*' + self.img_format))
        mask_list = glob.glob(m_path + '*' + self.img_format)
        total_number = len(mask_list)
        #imgs = np.ndarray((total_number, int(self.row), int(self.col)), dtype=np.uint8)
        color_imgs = np.ndarray((total_number, int(self.row), int(self.col), 3), dtype=np.uint8)
        msks = np.ndarray((total_number, int(self.row), int(self.col)), dtype=np.uint8)
        edgs = np.ndarray((total_number, int(self.row), int(self.col)), dtype=np.uint8)
        framename_list = list()
        
        for i in range(len(mask_list)):
            msks[i], edgs[i] = self.read_msk(mask_list[i])
            img_list = mask_list[i]
            img_name = img_list[len(m_path):]
            #Here, need adjust based on your dataset.
            framename_list.append(int(img_name[-7:-4]))
            color_imgs[i] = cv2.imread(r_path + img_name, cv2.IMREAD_COLOR)
            # ------------ below for self training
            #img_gt = cv2.imread(r_path + img_name, cv2.IMREAD_COLOR)
            #[row, col, colorchannels] = img_gt.shape
            #color_imgs[i] = img_gt[30:row-30, 30:col-30, :]
            
        # np.save(self.saved_folder + self.path + '.npy', framename_list)
        
        # added by junbong 11/13/2019
        all_files_list = os.listdir(r_path)
        image_name_list = [image_name for image_name in all_files_list if self.img_format in image_name]
        
        return color_imgs,msks,edgs,image_name_list
    
    
    def get_img_mask_list(self, root_image_path):
    
        train_img_list = glob.glob(root_image_path + '/images_train/' + '*' + self.img_format)
        valid_img_list = glob.glob(root_image_path + '/images_valid/' + '*' + self.img_format)
        test_img_list = glob.glob(root_image_path + '/images_test/' + '*' + self.img_format)
    
        train_mask_list = glob.glob(root_image_path + '/masks_train/' + '*' + self.img_format)
        valid_mask_list = glob.glob(root_image_path + '/masks_valid/' + '*' + self.img_format)
        test_mask_list = glob.glob(root_image_path + '/masks_test/' + '*' + self.img_format)
        
        img_list = train_img_list + valid_img_list + test_img_list
        mask_list = train_mask_list + valid_mask_list + test_mask_list
        
        assert len(img_list) == len(mask_list)
        total_number = len(mask_list)
        
        images = np.ndarray((total_number, int(self.row), int(self.col), 3), dtype=np.uint8)
        masks = np.ndarray((total_number, int(self.row), int(self.col), 3), dtype=np.uint8)
        
        for i in range(len(mask_list)):
            print('mask:', i)
            masks[i] = cv2.imread(mask_list[i], cv2.IMREAD_COLOR)
            
        for i in range(len(img_list)):
            print('img:', i)
            #img_name = os.path.relpath(img_path, root_image_path)
            #print(img_name)
            #all_image_names.append(img_name)
            images[i] = cv2.imread(img_list[i], cv2.IMREAD_COLOR)
            
        
        return images, masks, img_list, valid_img_list, test_img_list
    #==================================================================================
    #==================================================================================
    def get_train_test_indices(self, img_list, valid_image_names, test_image_names):
        
        imgsTrain_indices = [img_list.index(img_name) for img_name in img_list if img_name not in test_image_names and img_name not in valid_image_names]
        imgsValid_indices = [img_list.index(test_img_name) for test_img_name in valid_image_names]
        imgsTest_indices = [img_list.index(test_img_name) for test_img_name in test_image_names]

        return imgsTrain_indices, imgsValid_indices, imgsTest_indices
        
    
    def split_val(self, inputs, imgsTrain_indices, imgsVal_indices, imgsTest_indices):
        print('split_val')
        '''
        t_n = self.total_frames
        train_0, val_test_0, train_1, val_test_1, train_2, val_test_2 = train_test_split(inputs[0], inputs[1], inputs[2],
                                                                          test_size = t_n - self.n_frames_train, 
                                                                          random_state = self.random_seed)
        
        val_0, test_0, val_1, test_1, val_2, test_2 = train_test_split(val_test_0, val_test_1, val_test_2,
                                                                          test_size = t_n - self.n_frames_train - self.n_frames_val, 
                                                                          random_state = self.random_seed)
        '''                                               
        
        if isinstance(inputs[0], list):
            print('@@@@@@@@@@@@@@@@skip split val@@@@@@@@@@@@@@@@@@@@')
        else:
            
            images = inputs[0]
            masks = inputs[1]
            edges = inputs[2]
            
            train_0 = images[imgsTrain_indices]
            val_0 = images[imgsVal_indices]
            test_0 = images[imgsTest_indices]
            
            train_1 = masks[imgsTrain_indices]
            val_1 = masks[imgsVal_indices]
            test_1 = masks[imgsTest_indices]
            
            train_2 = edges[imgsTrain_indices]
            val_2 = edges[imgsVal_indices]
            test_2 = edges[imgsTest_indices]
            
            self.n_frames_train = train_0.shape[0]
            self.n_frames_val = val_0.shape[0]
            self.n_frames_test = test_0.shape[0]
            
            print(self.total_frames, self.n_frames_train, self.n_frames_val, self.n_frames_test)
            
            ''' to avoid errors that happen when validation set is 0
            split_train_val_0 = np.array_split(inputs[0], 4)
            train_0 = np.vstack((split_train_val_0[1], split_train_val_0[2], split_train_val_0[3]))
            test_0 = split_train_val_0[0]
            
            split_train_val_1 = np.array_split(inputs[1], 4)
            train_1 = np.vstack((split_train_val_1[1], split_train_val_1[2], split_train_val_1[3])) 
            test_1 = split_train_val_1[0]
            
            split_train_val_2 = np.array_split(inputs[2], 4)
            train_2 = np.vstack((split_train_val_2[1], split_train_val_2[2], split_train_val_2[3]))
            test_2 = split_train_val_2[0]
            '''
       
            
        return train_0, val_0, test_0, train_1, val_1, test_1, train_2, val_2, test_2
    #==================================================================================
    #==================================================================================
    def sample_loc(self, edge, number, on_edge = True, background = True):
        kernel = np.ones((int(self.output_size/2), int(self.output_size/2)), np.uint8)

        #smaller_edge = cv2.resize(edge, (500, 666), interpolation = cv2.INTER_CUBIC)
        #cv2.imshow('Show edge', smaller_edge)
        #cv2.waitKey(0)
        if on_edge:
            dilate_Edge = cv2.dilate(edge, kernel, iterations=1)
            loc = np.where( dilate_Edge > 0 )
        else:
            if background:
                loc = np.where( edge < 1 )
            else:
                loc = np.where( edge > 0 )
            

        index = np.argmax([len(np.unique(loc[0])), len(np.unique(loc[1])) ]) 
        sample_image_loc = np.random.choice(np.unique(loc[index]), number, replace = False)
        sample_pos = []
        for i in sample_image_loc:
            temp_index = np.where(loc[index] == i)[0]
            sample_pos.extend(np.random.choice(temp_index, 1)) # choose one index with a value randomly selected
            # since loc[0] and loc[1] are the array of indices that satisfy the condition, 
            # finding one random number a to later index using loc[0][a] loc[1][a] is enough
        
        
        return loc, sample_pos

        
    def crop_on_loc(self, inputs, loc, sample):
        image, mask, edge = inputs[0], inputs[1], inputs[2]
        imgs = np.ndarray((len(sample), int(self.input_size), int(self.input_size), 3), dtype=np.uint8)
        msks = np.ndarray((len(sample), int(self.input_size), int(self.input_size)), dtype=np.uint8)
        edgs = np.ndarray((len(sample), int(self.input_size), int(self.input_size)), dtype=np.uint8) 
    
        for i in range(len(sample)):
            imgs[i] = image[loc[0][sample[i]] :loc[0][sample[i]] + self.input_size, 
                            loc[1][sample[i]] :loc[1][sample[i]] + self.input_size, :]
            msks[i] =  mask[loc[0][sample[i]] :loc[0][sample[i]] + self.input_size, 
                            loc[1][sample[i]] :loc[1][sample[i]] + self.input_size]
            edgs[i] =  edge[loc[0][sample[i]] :loc[0][sample[i]] + self.input_size, 
                            loc[1][sample[i]] :loc[1][sample[i]] + self.input_size]
        return imgs, msks, edgs
        
    def crop_rand(self, inputs, edge_ratio = 0.18, background_ratio = .96):
        image, mask, edge = inputs[0], inputs[1], inputs[2]
        
        print('crop_rand')
        edge_pixel_num = np.count_nonzero(edge > 0)
        print(edge_pixel_num)
        # if number of edge pixels are less than the rand_crop_num, just sample from the background only
        if edge_pixel_num < self.rand_crop_num: 
            loc_p, sample_p = self.sample_loc(edge, 0, on_edge = True)
            loc_back, sample_back = self.sample_loc(mask, self.rand_crop_num, on_edge = False, background = True)
            loc_fore, sample_fore = self.sample_loc(mask, 0, on_edge = False, background = False)
        else:
            edge_number = int(self.rand_crop_num*edge_ratio)
            back_number = int((self.rand_crop_num - edge_number) * background_ratio)
            fore_number = int(self.rand_crop_num - edge_number - back_number)
            loc_p, sample_p = self.sample_loc(edge, edge_number, on_edge = True)
            loc_back, sample_back = self.sample_loc(mask, back_number, on_edge = False, background = True)
            loc_fore, sample_fore = self.sample_loc(mask, fore_number, on_edge = False, background = False)
        #pad and bias
        bound_in = int(np.ceil(self.input_size/2))
       
        image = np.lib.pad(image,((bound_in, bound_in), (bound_in, bound_in), (0,0)),'symmetric')
        mask = np.lib.pad(mask,((bound_in, bound_in), (bound_in, bound_in)),'symmetric')
        edge = np.lib.pad(edge,((bound_in, bound_in), (bound_in, bound_in)),'symmetric')
        
        imgs_p, msks_p, edgs_p = self.crop_on_loc([image, mask, edge], loc_p, sample_p)
        imgs_back, msks_back, edgs_back = self.crop_on_loc([image, mask, edge], loc_back, sample_back)
        imgs_fore, msks_fore, edgs_fore = self.crop_on_loc([image, mask, edge], loc_fore, sample_fore)
        # return np.r_[imgs_p, imgs_n], np.r_[msks_p, msks_n], np.r_[edgs_p, edgs_n]
        return np.r_[imgs_p, imgs_back, imgs_fore], np.r_[msks_p, msks_back, msks_fore], np.r_[edgs_p, edgs_back, edgs_fore]
    #==================================================================================
    #==================================================================================
    def pad_img(self, inputs):
        num_y = int(np.ceil(self.col/self.output_size));
        num_x = int(np.ceil(self.row/self.output_size));
        sym = int(np.ceil(self.input_size/2 - self.output_size/2))
        for i in range(3):
            inputs[i] = np.lib.pad(inputs[i], ((0, int(num_x*self.output_size - inputs[i].shape[0])),(0, int(num_y*self.output_size - inputs[i].shape[1]))), 'symmetric')
            inputs[i] = np.lib.pad(inputs[i], ((sym, sym), (sym, sym)),'symmetric');
        return inputs[0], inputs[1], inputs[2]
    
    def crp_e(self, inputs):
        num_y = int(np.ceil(self.col/self.output_size));
        num_x = int(np.ceil(self.row/self.output_size));
        
        
        imgCrop = np.ndarray((num_x*num_y, int(self.input_size), int(self.input_size)), dtype=np.uint8)
        mskCrop = np.ndarray((num_x*num_y, int(self.input_size), int(self.input_size)), dtype=np.uint8)
        edgCrop = np.ndarray((num_x*num_y, int(self.input_size), int(self.input_size)), dtype=np.uint8)
 
        for row in range(num_y):
            for col in range(num_x):
                imgCrop[col*num_y+row] = inputs[0][col*self.output_size:col*self.output_size+self.input_size, row*self.output_size:row*self.output_size+self.input_size]              
                mskCrop[col*num_y+row] = inputs[1][col*self.output_size:col*self.output_size+self.input_size, row*self.output_size:row*self.output_size+self.input_size]
                edgCrop[col*num_y+row] = inputs[2][col*self.output_size:col*self.output_size+self.input_size, row*self.output_size:row*self.output_size+self.input_size]  
                
        return imgCrop, mskCrop, edgCrop
                
    def crop_even(self, inputs):
        image, mask, edge = self.pad_img(inputs)
        imgCrop, mskCrop, edgCrop = self.crp_e([image, mask, edge])
        return imgCrop, mskCrop, edgCrop
    #==================================================================================
    #==================================================================================
    def crop_train(self, n_frames, image, mask, edge):
        imgs_r = np.ndarray((n_frames*self.rand_crop_num, int(self.input_size), int(self.input_size), 3), dtype=np.uint8)
        msks_r = np.ndarray((n_frames*self.rand_crop_num, int(self.input_size), int(self.input_size)), dtype=np.uint8)
        edgs_r = np.ndarray((n_frames*self.rand_crop_num, int(self.input_size), int(self.input_size)), dtype=np.uint8)
        
        for i in range(n_frames):
            imgs_r[i*self.rand_crop_num:(i+1)*self.rand_crop_num], msks_r[i*self.rand_crop_num:(i+1)*self.rand_crop_num], edgs_r[i*self.rand_crop_num:(i+1)*self.rand_crop_num] = self.crop_rand([image[i], mask[i], edge[i]])

        return imgs_r, msks_r, edgs_r
    

    def get_names_in_folder(self, folder_path):
        image_files = [f for f in os.listdir(folder_path) if os.path.isfile(folder_path + '/' + f) and f.endswith('.png')]
        return image_files

    
    def save_split_into_text(self, saved_file_path, img_list, imgsTrain_index, imgsVal_index, imgsTest_index):
        f= open(saved_file_path,"w+")

        imgTrain_name_list = [img_list[i] for i in imgsTrain_index]
        imgVal_name_list = [img_list[i] for i in imgsVal_index]
        imgTest_name_list = [img_list[i] for i in imgsTest_index]

        f.write("{}\n".format(', '.join(imgTrain_name_list)))
        f.write("{}\n".format(', '.join(imgVal_name_list)))
        f.write("{}\n".format(', '.join(imgTest_name_list)))
        
        imgsTrain_index_string = ' '.join(str(e) for e in imgsTrain_index)
        imgsVal_index_string = ' '.join(str(e) for e in imgsVal_index)
        imgsTest_index_string = ' '.join(str(e) for e in imgsTest_index)
        
        f.write("imgsTrain_index: {}\n".format(imgsTrain_index_string))
        f.write("imgsVal_index: {}\n".format(imgsVal_index_string))
        f.write("imgsTest_index: {}\n".format(imgsTest_index_string))
        f.close()
        print("written to file")
    
    
    def crop(self, saved_file_path, root_image_path, crop_type):
    
        #train_image_names = self.get_names_in_folder(train_image_path)
        #valid_image_names = self.get_names_in_folder(valid_image_path)
        #test_image_names = self.get_names_in_folder(test_image_path)
        
        #all_image_names = train_image_names + valid_image_names + test_image_names
        
        # image,mask,edge,img_list = self.r_img_msk()
        image, mask, all_image_names, valid_image_names, test_image_names = self.get_img_mask_list(root_image_path)
        edge = mask
        print('img list length: ', len(all_image_names))
        
        imgsTrain_index, imgsVal_index, imgsTest_index = self.get_train_test_indices(all_image_names, valid_image_names, test_image_names)
        print('train images number: ',len(imgsTrain_index))
        print('valid images number: ',len(imgsVal_index))
        print('test images number: ',len(imgsTest_index))
        
        self.save_split_into_text(saved_file_path, all_image_names, imgsTrain_index, imgsVal_index, imgsTest_index)
        
        imgsTrain, imgsVal, imgsTest, msksTrain, msksVal, msksTest, edgsTrain, edgsVal, edgsTest = self.split_val([image, mask, edge], imgsTrain_index, imgsVal_index, imgsTest_index)
        # ----------- 8/18/2020 Chnage below line to crop train or validation images
        if crop_type == 'train':
            imgs_train,msks_train,edgs_train = self.crop_train(self.n_frames_train, imgsTrain, msksTrain, edgsTrain)
        elif crop_type == 'valid':
            imgs_train,msks_train,edgs_train = self.crop_train(self.n_frames_val, imgsVal, msksVal, edgsVal)
        else:
            print('Crop Type Error: ', crop_type)
        avg = np.mean(imgsTrain)
        std = np.std(imgsTrain)
        
        return imgs_train, msks_train, edgs_train, avg, std, imgsTrain_index, imgsVal_index, imgsTest_index
