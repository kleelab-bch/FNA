import numpy as np
import os
import cv2
import glob
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class DataPreparer:
    CROP_SIZE = 128
    MARGIN = 30         # (128 - 68) / 2
    SPLIT_RATE = 0.2    # val : train = 2 : 8

    def __init__(self, im_path, mask_path=None, crop_num=200, batch_size=32):
        self.im_path = im_path
        self.mask_path = mask_path
        self.crop_num = crop_num
        self.imgs = []
        self.masks = []
        self.edges = []
        self.img_list = []      # for file names
        self.mask_list = []
        self.batch_size = batch_size
        self.num_train = None
        self.num_val = None

    def load_img(self):
        """load images and masks from disk and get edges"""
        self.img_list = sorted(glob.glob(os.path.join(self.im_path, '*.png')))

        if len(self.img_list) == 0:
            raise ValueError('there is no matching file in ' + self.im_path)
        if self.mask_list:
            assert len(self.img_list) == len(self.mask_list), 'inconsistent number of imgs and masks'

    def load_mask(self):
        self.mask_list = sorted(glob.glob(os.path.join(self.mask_path, '*.png')))

        if len(self.mask_list) == 0:
            raise ValueError('there is no matching file in ' + self.mask_path)
        if self.img_list:
            assert len(self.img_list) == len(self.mask_list), 'inconsistent number of imgs and masks'

    def get_edge(self, mask):
        """detect edges from input mask"""
        mask[mask > 0.5] = 255
        edg = cv2.Canny(mask, 50, 100)
        kernel = np.ones((3, 3), np.uint8)
        edg = cv2.dilate(edg, kernel, iterations=3)     # needs tuning
        return edg

    def crop_on_loc(self, inp, rows, cols):
        """crop given input at rows and cols, cropping size: CROP_SIZE x CROP_SIZE"""
        out = []
        offset0 = int(self.CROP_SIZE / 2)
        offset1 = self.CROP_SIZE - offset0
        for row, col in zip(rows, cols):
            out.append(inp[row - offset0:row + offset1, col - offset0:col + offset1])
        out = np.stack(out, axis=0)
        return out

    def sample_loc(self, edge, number, on_edge=True):
        if on_edge:
            loc = np.where(edge > 0)      # a tuple of two arrays, represent indices of row and col respectively
        else:
            loc = np.where(edge < 1)
        sample_idx = np.random.choice(np.arange(len(loc[0])), size=number, replace=False)  # guarantee uniqueness
        return loc[0][sample_idx], loc[1][sample_idx]

    def get_stats(self):
        """get mean and std of training set by sampling"""
        num_samples = len(self.img_list) if len(self.img_list) < 10 else 10
        sub_img_list = np.random.choice(self.img_list, size=num_samples, replace=False)
        imgs = []
        for img_name in sub_img_list:
            imgs.append(cv2.imread(img_name, 0))

        im_mean = np.mean(imgs)
        im_std = np.std(imgs)
        return im_mean, im_std

    def crop_all(self):
        edge_ratio = 0.6
        edge_num = int(self.crop_num * edge_ratio)      # number of crops centered at edges of the cell
        other_num = self.crop_num - edge_num
        pad_width = int(np.ceil(self.CROP_SIZE / 2))

        for img_name, mask_name in zip(self.img_list, self.mask_list):
            img = cv2.imread(img_name, 0).astype('float32')
            mask = cv2.imread(mask_name, 0)
            edge = self.get_edge(mask)

            # sampling crop centers
            row_p, col_p = self.sample_loc(edge, edge_num, True)
            row_p += np.random.randint(-10, 11, edge_num)       # add some random offsets to locations on edge
            col_p += np.random.randint(-10, 11, edge_num)
            row_n, col_n = self.sample_loc(edge, other_num, False)
            rows = np.hstack((row_p, row_n)) + pad_width
            cols = np.hstack((col_p, col_n)) + pad_width

            img = np.pad(img, pad_width, 'symmetric')
            mask = np.pad(mask, pad_width, 'symmetric')
            edge = np.pad(edge, pad_width, 'symmetric')

            self.imgs.append(self.crop_on_loc(img, rows, cols))
            self.masks.append(self.crop_on_loc(mask, rows, cols))
            self.edges.append(self.crop_on_loc(edge, rows, cols))

            # # Visualization
            # f1 = plt.subplot(311)
            # plt.imshow(self.imgs[-1][0], 'gray')
            # plt.title(img_name)
            # f2 = plt.subplot(312)
            # plt.imshow(self.masks[-1][0], 'gray')
            # plt.title(mask_name)
            # f3 = plt.subplot(313)
            # plt.imshow(self.edges[-1][0], 'gray')
            # plt.show()

        # normalization
        img_mean, img_std = self.get_stats()
        np.savez(self.im_path + '/train_mean_std.npz', mean=img_mean, std=img_std)
        self.imgs = (self.imgs - img_mean) / img_std

        self.imgs = np.concatenate(self.imgs, axis=0)
        self.masks = self.crop_margin(self.masks)
        self.edges = self.crop_margin(self.edges)
        return

    def crop_margin(self, inp):
        """cut out given MARGIN from the inp"""
        inp = np.concatenate(inp, axis=0).astype('float32')
        return inp[:, self.MARGIN:-self.MARGIN, self.MARGIN:-self.MARGIN]

    def binarize_mask_edge(self):
        # need the operand to be float type
        self.masks /= 255.
        self.edges /= 255.
        return

    def get_generator(self):
        """get generators for training set and validation set"""
        generator = ImageDataGenerator(rotation_range=90,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       fill_mode='reflect')
        # preprocessing
        self.binarize_mask_edge()

        self.imgs = self.add_axis(self.imgs, repeat=True)
        self.masks = self.add_axis(self.masks)
        self.edges = self.add_axis(self.edges)
        seed = 66

        # split crops
        imgs_tr, imgs_val, masks_tr, masks_val, edges_tr, edges_val\
            = train_test_split(self.imgs, self.masks, self.edges,
                               test_size=self.SPLIT_RATE, random_state=seed)
        self.num_train = len(imgs_tr)
        self.num_val = len(imgs_val)

        # feed generator with the corresponding data
        gene_img = generator.flow(imgs_tr, batch_size=self.batch_size, seed=seed)
        gene_mask = generator.flow(masks_tr, batch_size=self.batch_size, seed=seed)
        gene_edge = generator.flow(edges_tr, batch_size=self.batch_size, seed=seed)
        out_gene = zip(gene_mask, gene_edge)
        train_generator = zip(gene_img, out_gene)

        # # Visualize augmented crops for debugging
        # for i in range(10):
        #     img = gene_img.next()[i][:, :, 0]
        #     mask = gene_mask.next()[i][:, :, 0]
        #     f1 = plt.subplot(211)
        #     plt.imshow(img, 'gray')
        #     f2 = plt.subplot(212)
        #     plt.imshow(mask, 'gray')
        #     plt.show()

        gene_img = generator.flow(imgs_val, batch_size=self.batch_size, seed=seed)
        gene_mask = generator.flow(masks_val, batch_size=self.batch_size, seed=seed)
        gene_edge = generator.flow(edges_val, batch_size=self.batch_size, seed=seed)
        out_gene = zip(gene_mask, gene_edge)
        val_generator = zip(gene_img, out_gene)

        return train_generator, val_generator

    def add_axis(self, img, repeat=False):
        img = img[..., np.newaxis]
        return img if not repeat else img.repeat(3, axis=-1)

    def get_mean(self, imgs):
        return np.mean(imgs)

    def get_std(self, imgs):
        return np.std(imgs)

    def save_mean_std(self):
        pass

    def to_grey(self, savepath):
        if not self.img_list:
            raise ValueError("img_list is empty")
        else:
            # plt.figure()
            for name in self.img_list:
                img = cv2.imread(name, 0)
                img = cv2.resize(img, (1128, 832))
                # img = self.normalize_grey(img, stats_file)
                filename = os.path.splitext(os.path.basename(name))[0] + '.png'
                cv2.imwrite(os.path.join(savepath, filename), img)

                # plt.imshow(img, 'gray')
                # plt.show()
            print('to_grey done')

    def to_white(self, source):
        # set mask values to 255
        source = self.img_list if source == 'img' else self.mask_list
        if not source:
            raise ValueError(str(source) + " is empty")
        else:
            for name in source:
                img = cv2.imread(name, 0)
                img = cv2.resize(img, (1128, 832))
                img[img > 0] = 255

                filename = os.path.splitext(os.path.basename(name))[0] + '.png'
                cv2.imwrite(os.path.join(out_path, filename), img)

    def main(self):
        self.load_img()
        self.load_mask()
        self.crop_all()
        gene = self.get_generator()
        return gene


if __name__ == '__main__':
    # initialization
    img_path = 'DataSet_label/Human_Muscle_PF573228/sample_test'
    mask_path = 'DataSet_label/Human_Muscle_PF573228/sample_test'

    out_path = 'DataSet_label/Human_Muscle_PF573228/sample_test_result'
    # stats_path = 'DataSet_label/FAK_N1/train/train_mean_std.npz'

    ob = DataPreparer(img_path, None)

    # process raw images
    # ob.load_img()
    # ob.to_grey(out_path)      # if necessary
    # ob.to_white('img')

    # process masks
    ob.load_mask()
    ob.to_white('mask')


