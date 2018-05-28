import torch.utils.data as data
import os
import numpy as np
import cv2
#/mnt/lustre/share/dingmingyu/new_list_lane.txt

class MyDataset(data.Dataset):
    def __init__(self, file, dir_path, new_width, new_height, label_width, label_height):
        imgs = []
        fw = open(file, 'r')
        lines = fw.readlines()
        for line in lines:
            words = line.strip().split()
            imgs.append((words[0], words[1]))
        self.imgs = imgs
        self.dir_path = dir_path
        self.height = new_height
        self.width = new_width
        self.label_height = label_height
        self.label_width = label_width

    def __getitem__(self, index):
        path, label = self.imgs[index]
        path = os.path.join(self.dir_path, path)
        img = cv2.imread(path).astype(np.float32)
        img = img[:,:,:3]
        img = cv2.resize(img, (self.width, self.height))
        img -= [104, 117, 123]
        img = img.transpose(2, 0, 1)
        gt = cv2.imread(label,-1)
        gt = cv2.resize(gt, (self.label_width, self.label_height), interpolation = cv2.INTER_NEAREST)  
        if len(gt.shape) == 3:
            gt = gt[:,:,0]

        gt_num_list = list(np.unique(gt))
        gt_num_list.remove(0)
        target_ins = np.zeros((4, gt.shape[0],gt.shape[1])).astype('uint8')
        for index, ins in enumerate(gt_num_list):
            target_ins[index,:,:] += (gt==ins)
        return img, target_ins, len(gt_num_list)

    def __len__(self):
        return len(self.imgs)
