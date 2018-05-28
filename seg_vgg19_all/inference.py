import shutil
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import lanenet
import os
import torchvision as tv
from torch.autograd import Variable
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--img_list', dest='img_list', default='/mnt/lustre/dingmingyu/data/test_pic/list.txt',                         help='the test image list', type=str)
parser.add_argument('--img_dir', dest='img_dir', default='/mnt/lustre/dingmingyu/data/test_pic/',
                        help='the test image dir', type=str)
parser.add_argument('--model_path', dest='model_path', default='checkpoints/004_checkpoint.pth.tar',
                        help='the test model', type=str)

def main():
    global args
    args = parser.parse_args()
    print ("Build model ...")
    model = lanenet.Net()
    model = torch.nn.DataParallel(model).cuda()
    state = torch.load(args.model_path)['state_dict']
    model.load_state_dict(state)
    model.eval()    

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    f = open(args.img_list)
    ni = 0
    for line in f:
        line = line.strip()
        arrl = line.split(" ") 
        image = cv2.imread(args.img_dir + arrl[0])
        image = cv2.resize(image,(833,705)).astype(np.float32)
        image -= [104, 117, 123]
        image = image.transpose(2, 0, 1)
        #image = cv2.imread(args.img_dir + arrl[0], -1)
        #image = cv2.resize(image, (732,704), interpolation = cv2.INTER_NEAREST)
        print image.shape
        image = torch.from_numpy(image).unsqueeze(0)
        image = Variable(image.float().cuda(0), volatile=True)
    	output = model(image)
    	#prob = output.data[0].max(0)[1].cpu().numpy()
        #print prob.max(),prob.shape
        output = torch.nn.functional.softmax(output[0],dim=0)
        prob = output.data.cpu().numpy()
#       prob = output.data[0].max(0)[1].cpu().numpy()
#        print output.size()
        #print output.max(),type(output)

        print prob[1].max(),prob.shape
        prob = (prob[1] >= 0.3)

#        output = torch.nn.functional.softmax(output[0],dim=0)
#        prob = output.data.cpu().numpy() 
#        print prob[1].max(),prob[2].max(),prob[3].max(),prob[4].max(),prob.shape 

        probAll = np.zeros((prob.shape[0], prob.shape[1], 3), dtype=np.float)
        probAll[:,:,0] += prob # line 1
        probAll[:,:,1] += prob # line 2
        probAll[:,:,2] += prob # line 3


        probAll = np.clip(probAll * 255, 0, 255)

        test_img = cv2.imread(args.img_dir + arrl[0], -1)
        probAll = cv2.resize(probAll, (1280,720), interpolation = cv2.INTER_NEAREST)
        test_img = cv2.resize(test_img, (1280,720))

        ni = ni + 1
        test_img = np.clip(test_img + probAll, 0, 255).astype('uint8')
        cv2.imwrite(args.img_dir + 'prob/test_' + str(ni) + '_lane.png', test_img)
        print('write img: ' + str(ni+1))
    f.close()

if __name__ == '__main__':
    main()

