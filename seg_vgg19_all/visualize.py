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



print ("Build model ...")
model = lanenet.Net()
model = torch.nn.DataParallel(model).cuda()
state = torch.load('checkpoints/020_checkpoint.pth.tar')['state_dict']
model.load_state_dict(state)
model.eval()

state.keys()
state['module.bottom.conv1.cbr_unit.0.weight']

import numpy as np
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp((x)*50)
    return e_x / e_x.sum((1,2))

%matplotlib inline
import matplotlib.pyplot as plt
import cv2
from PIL import Image 
a = np.zeros((3,21,21))
for index, item in enumerate(state['module.bottom.conv1.cbr_unit.0.weight'].cpu().numpy()):
    i = index/4
    j = index % 4
    a[:,i*6:i*6 +3,j*6:j*6+3] = (softmax(item)*255).astype('uint8')
    print (softmax(item)*255).astype('uint8')
plt.imshow(a.transpose(1,2,0))
cv2.imwrite('all.png',a.transpose(1,2,0))
    #print softmax(item)
    #plt.imshow((softmax(item)*255).astype('uint8'))
    #plt.savefig("%d.jpg" % index)  

a = cv2.imread('all.png')
plt.imshow(a[:,:,::-1])
