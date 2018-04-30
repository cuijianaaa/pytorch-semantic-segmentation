import os

import numpy as np
import torch
from PIL import Image
from torch.utils import data
import random
import math
num_classes = 19
ignore_label = 255
root = '/media/cj/Elements/cityscapes_ins'


palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


def make_dataset(quality, mode):
    assert (quality == 'fine' and mode in ['train', 'val']) or \
           (quality == 'coarse' and mode in ['train', 'train_extra', 'val'])

    if quality == 'coarse':
        img_dir_name = 'leftImg8bit_trainextra' if mode == 'train_extra' else 'leftImg8bit_trainvaltest'
        mask_path = os.path.join(root, 'gtCoarse', 'gtCoarse', mode)
        mask_postfix = '_gtCoarse_labelIds.png'
    else:
        img_dir_name = 'leftImg8bit_trainvaltest'
        mask_path = os.path.join(root, 'gtFine_trainvaltest', 'gtFine', mode)
        mask_postfix = '_gtFine_labelIds.png'
    img_path = os.path.join(root, img_dir_name, 'leftImg8bit', mode)
    assert os.listdir(img_path) == os.listdir(mask_path)
    items = []
    categories = os.listdir(img_path)
    for c in categories:
        c_items = [name.split('_leftImg8bit.png')[0] for name in os.listdir(os.path.join(img_path, c))]
        for it in c_items:
            item = (os.path.join(img_path, c, it + '_leftImg8bit.png'), os.path.join(mask_path, c, it + mask_postfix))
            items.append(item)
    return items


class CityScapes(data.Dataset):
    def __init__(self, quality, mode, joint_transform=None, sliding_crop=None, transform=None, target_transform=None):
        self.imgs = make_dataset(quality, mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.quality = quality
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
        mask = np.array(mask)
        mask_copy = mask.copy()
        for k, v in self.id_to_trainid.items():
            mask_copy[mask == k] = v
        mask = Image.fromarray(mask_copy.astype(np.uint8))

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        if self.sliding_crop is not None:
            img_slices, mask_slices, slices_info = self.sliding_crop(img, mask)
            if self.transform is not None:
                img_slices = [self.transform(e) for e in img_slices]
            if self.target_transform is not None:
                mask_slices = [self.target_transform(e) for e in mask_slices]
            img, mask = torch.stack(img_slices, 0), torch.stack(mask_slices, 0)
            return img, mask, torch.LongTensor(slices_info)
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                mask = self.target_transform(mask)
            return img, mask

    def __len__(self):
        return len(self.imgs)

def make_instance(mode):
    assert (mode in ['train', 'val'])
           
    img_dir_name = 'leftImg8bit_trainvaltest'
    mask_path = os.path.join(root, 'gtFine_trainvaltest', 'gtFine', mode)
    mask_postfix = '_gtFine_instanceIds.png'
    img_path = os.path.join(root, img_dir_name, 'leftImg8bit', mode)
    assert os.listdir(img_path) == os.listdir(mask_path)
    items = []
    categories = os.listdir(img_path)
    for c in categories:
        c_items = [name.split('_leftImg8bit.png')[0] for name in os.listdir(os.path.join(img_path, c))]
        for it in c_items:
            item = (os.path.join(img_path, c, it + '_leftImg8bit.png'), os.path.join(mask_path, c, it + mask_postfix))
            items.append(item)
    return items

class Instance(data.Dataset):
    def __init__(self, mode, joint_transform=None, sliding_crop=None, transform=None, target_transform=None):
        self.imgs = make_instance(mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        #self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
        #                      3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
        #                      7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
        #                      14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
        #                      18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
        #                      28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
        self.id_to_trainid = {
            -1: 0,#ignore_label,#
            1: 0, #ignore_label, #//ego vehicle
            2: 0,#ignore_label, #//rectification border
            3: 0,#ignore_label, #//out of roi
            4: 0,#ignore_label, #//static
            5: 0,#ignore_label, #//dynamic
            6: 0,#ignore_label, #//ground
            7: 0,            #//road
            8: 0,#1,            #//sidewalk
            9: 0,#ignore_label, #//parking
            10:0,# ignore_label,#//rail track
            11: 0,#2,           #//building
            12: 0,#3,           #//wall
            13: 0,#4,           #//fence
            14: 0,#ignore_label,#//guard rail
            15: 0,#ignore_label,#//bridge
            16: 0,#ignore_label,#//tunnel
            17: 0,#5,           #//pole
            18: 0,#ignore_label,#//group pole
            19: 0,#6,           #//traffic light
            20: 0,#7,           #//traffic sign
            21: 0,#8,           #//tree glass...
            22: 0,#9,           #//dixing
            23: 0,#10,          #//sky
            24: 0,#11,          #//person
            25: 0,#12,          #//rider
            26: 13,          #//car
            27: 0,#14,          #//truck
            28: 0,#15,          #//bus
            29: 0,#ignore_label,#//caravan
            30: 0,#ignore_label,#//trailer
            31: 0,#16,          #//train
            32: 0,#17,          #//motorcycle
            33: 0}#18}          #//becycle

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)

        mask = np.array(mask)
        mask_class = mask / 1000
        mask_class = mask_class.astype(np.uint8) 
        mask_ins = mask % 1000
        mask_ins = mask_ins.astype(np.uint8)
        #print 'class max', np.max(mask_class)
        #print  'class min', np.min(mask_class)
        #print 'ins max', np.max(mask_instance)
        #print 'ins min', np.min(mask_instance)
        #print mask.dtype 
        mask_class_copy = mask_class.copy()
       
        for k, v in self.id_to_trainid.items():
            mask_class_copy[mask_class == k] = v
        mask_class = Image.fromarray(mask_class_copy)
        mask_ins = Image.fromarray(mask_ins)
        #img = np.array(img)
        #img3c = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        #img3c[:,:,0] = mask_class_copy
        #img3c[:,:,1] = mask_class_copy
        #img3c[:,:,2] = mask_class_copy
        #img = Image.fromarray(img3c) 
        if self.joint_transform is not None:
            img, mask_class, mask_ins = self.joint_transform(img, mask_class, mask_ins)
        if self.sliding_crop is not None:
            img_slices, mask_class_slices, mask_ins_slices, slices_info = self.sliding_crop(img, mask_class, mask_ins)
            #print img_slices
            if self.transform is not None:
                img_slices = [self.transform(e) for e in img_slices]
            if self.target_transform is not None:
                mask_class_slices = [self.target_transform(e) for e in mask_class_slices]
                #i = 0
                #print 'before'
                #for e in mask_class_slices:
                #    print i,' size ', e
                #    i = i + 1
                mask_ins_slices = [self.target_transform(e) for e in mask_ins_slices]
                #i = 0
                #print 'end'
                #for e in mask_ins_slices:
                #    print i,' size ', e
                #    i = i + 1
            img, mask, ins = torch.stack(img_slices, 0), torch.stack(mask_class_slices, 0), torch.stack(mask_ins_slices, 0)
            #print 'img ', img.size()
            #print 'mask ', mask.size()
            #print 'ins ', ins.size()
            return img, mask, ins, torch.LongTensor(slices_info)
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                mask_class = self.target_transform(mask_class)
                mask_ins = self.target_transform(mask_ins)
            return img, mask_class, mask_ins

    def __len__(self):
        return len(self.imgs)

class InstanceGenData(data.Dataset):
    def __init__(self, mode, joint_transform=None, transform=None, target_transform=None):
        self.imgs = make_instance(mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
        self.grid_size = 50
        self.img_size = 384
        self.grid_num = 10
    def __getitem__(self, index):

        img = np.zeros((self.img_size, self.img_size, 3), dtype = np.uint8)
        mask = np.zeros((self.img_size, self.img_size), dtype = np.uint8)
        for i in range(self.img_size):
            img[i,:,1] = i
            img[:,i,2] = i
        record = -1
        rc = []
        idx = 1
        for i in range(self.grid_num):
            r = random.randint(self.grid_size, self.img_size-self.grid_size)
            c = random.randint(self.grid_size, self.img_size-self.grid_size)
            if((r * self.img_size + c) > record):
                have = 0
                for rr,cc in rc:
                    if((((rr-r)**2 + (cc-c)**2)**0.5) <= ((2 * self.grid_size + 2) * math.sqrt(2))):
                        have = 1
                        break
                if(not have):
                    rc.append([r,c])
                    record = r * self.img_size + c
                    img[(r-self.grid_size):(r+self.grid_size), (c-self.grid_size):(c+self.grid_size), 0] = 255
                    mask[(r-self.grid_size+1):(r+self.grid_size-1), (c-self.grid_size+1):(c+self.grid_size-1)] = 1 #idx
                    mask[[(r-self.grid_size),(r+self.grid_size)], [(c-self.grid_size),(c+self.grid_size)]] = 2
                    idx = idx + 1

        img = Image.fromarray(img)
        mask_class = Image.fromarray(mask)
        mask_ins = Image.fromarray(mask)

        if self.joint_transform is not None:
            img, mask_class, mask_ins = self.joint_transform(img, mask_class, mask_ins)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask_class = self.target_transform(mask_class)
            mask_ins = self.target_transform(mask_ins)
        return img, mask_class, mask_ins

    def __len__(self):
        return 100
