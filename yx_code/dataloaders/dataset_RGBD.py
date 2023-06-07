import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
from PIL import Image
from os import listdir
from os.path import splitext
class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        #logging.info('Creating dataset with {len(self.ids)} examples')
        if self.split == 'train':
            self.imgs_dir= self._base_dir +"/train/img/"
            self.masks_dir= self._base_dir +"/train/mask/"
            self.hsm_dir = self._base_dir + "/train/dsm/"
            self.sample_list = [splitext(file)[0] for file in listdir(self.imgs_dir)
                    if not file.startswith('.')]
        elif self.split == 'val':
           self.imgs_dir= self._base_dir +"/val/img/"
           self.masks_dir= self._base_dir +"/val/mask/"
           self.hsm_dir = self._base_dir + "/val/dsm/"
           self.sample_list = [splitext(file)[0] for file in listdir(self.imgs_dir)
                    if not file.startswith('.')]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        
        mask_file = glob(self.masks_dir + case +  '.*')
        img_file = glob(self.imgs_dir + case + '.*')
        hsm_file = glob(self.hsm_dir + case + '.*')
        #label = Image.open(mask_file[0])
        #image = Image.open(img_file[0])
        image = cv2.imread(img_file[0],3)
        # image = image / 255.0
        image = RGB01(image)
        hsm = cv2.imread(hsm_file[0], -1)
        hsm = hsm.reshape((256,256,1))
        image = np.concatenate((image, hsm), axis=2)
        if mask_file!=[]:
            label = cv2.imread(mask_file[0],cv2.IMREAD_GRAYSCALE)
            # if np.max(label)>=5:
                # print("11")
                # print(str(np.max(label)))
            label[label >= 7] = 0
            label[label <= 0] = 0
            #label[label>0]=1
            #print(label.shape)
        else:
            label=np.empty((image.shape[0],image.shape[1]))
            #print(label.shape)
        
        #print(label.shape)       

        sample = {'image': image, 'label': label}
        if self.split == "train":
            sample = self.transform(sample)
        #if self.split == "val":
        #    sample = self.transform(sample)
        sample["idx"] = idx
        return sample

def RGB01(img):
    img = img.astype('int16')
    r = img[:,:,0:1]
    g = img[:, :, 1:2]
    b = img[:, :, 2:3]
    r01 = r.copy()
    g01 = g.copy()
    b01 = b.copy()
    rgb = r+ g +b
    r01 = r01/(rgb+0.000001)
    g01 = g01/(rgb+0.000001)
    b01 = b01 / (rgb+0.000001)
    img_rgb01 = np.concatenate((r01,g01,b01),axis = 2)
    return img_rgb01

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        #if random.random() > 0.5:
        #    image, label = random_rot_flip(image, label)
        #elif random.random() > 0.5:
        #    image, label = random_rotate(image, label)
        x, y = image.shape[0],image.shape[1]
        #print(image.shape)
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y,1), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        #print(image.shape)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        image = torch.from_numpy(image)
        label = torch.from_numpy(label.astype(np.uint8))
        #print(image.shape)
        sample = {'image': image, 'label': label}
        
        return sample


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
