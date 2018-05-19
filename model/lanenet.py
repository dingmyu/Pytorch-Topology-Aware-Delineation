#--coding:utf-8--
import torch.nn as nn
import torch
import math

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, stride=1, k_size=3, padding=1, bias=True, dilation=1, with_bn=True):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                            padding=padding, stride=stride, bias=bias, dilation=dilation)

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),
                                          nn.ReLU(inplace=True),)
        else:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class sharedBottom(nn.Module):
    def __init__(self,):
        super(sharedBottom, self).__init__()
        self.conv1 = conv2DBatchNormRelu(4, 16, 2) 
        self.conv2a1 = conv2DBatchNormRelu(16, 16, 2)
        self.conv2a2 = conv2DBatchNormRelu(16,8)
        self.conv2a3 = conv2DBatchNormRelu(8,4)
        self.conv2a4 = conv2DBatchNormRelu(4,4)
        self.conv2a_strided = conv2DBatchNormRelu(32,32,2)
        self.conv3 = conv2DBatchNormRelu(32,32,2)
        self.conv4 = conv2DBatchNormRelu(32,32,1)
        self.conv6 = conv2DBatchNormRelu(32,64,2)
        self.conv8 = conv2DBatchNormRelu(64,64,1)
        self.conv9 = conv2DBatchNormRelu(64,128,2)
        self.conv11 = conv2DBatchNormRelu(128,128,1)
        self.conv11_1 = conv2DBatchNormRelu(128,32,1)
        self.conv11_2 = conv2DBatchNormRelu(128,32,1)
        self.conv11_3 = conv2DBatchNormRelu(128,32,1)
        self.conv11_4 = conv2DBatchNormRelu(32,64,1)
        self.conv11_6 = conv2DBatchNormRelu(32,64,1)
        self.conv11_5 = conv2DBatchNormRelu(64,128,1)

    def forward(self, x, y):
        y = nn.Upsample(size=(705,833),mode='bilinear')(y)
        x = torch.cat([x, y], dim = 1)
        x = self.conv1(x)
        x1 = self.conv2a1(x)
        x2 = self.conv2a2(x1)
        x3 = self.conv2a3(x2)
        x4 = self.conv2a4(x3)
        x = torch.cat([x1, x2, x3, x4], dim = 1)
        x = self.conv2a_strided(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv6(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv11(x)
        x1= self.conv11_1(x)
        x2= self.conv11_2(x)
        x3= self.conv11_3(x)
        x4= self.conv11_4(x3)
        x6= self.conv11_6(x2)
        x5= self.conv11_5(x4)
        x = torch.cat([x1, x5, x6], dim = 1)
        return x

class laneNet(nn.Module):
    def __init__(self,):
        super(laneNet, self).__init__()
        self.conv11_7 = conv2DBatchNormRelu(224,128,1)
        self.conv11_8 = conv2DBatchNormRelu(128,128,1)
        self.conv11_9 = conv2DBatchNormRelu(128,128,1)
        self.conv12 = conv2DBatchNormRelu(128,16,1)
        self.conv13 = conv2DBatchNormRelu(16,8,1)
        self.conv14 = nn.Conv2d(8, 2, 1,stride = 1,padding = 0, bias=True)
    def forward(self, x):
        x = self.conv11_7(x)
        x = self.conv11_8(x)
        x = self.conv11_9(x)
        x = nn.Upsample(size=(45,53),mode='bilinear')(x)
        x = self.conv12(x)
        x = nn.Upsample(size=(177,209),mode='bilinear')(x)
        x = self.conv13(x)
        x = self.conv14(x)
        return x        


class Net(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        # 下式等价于nn.Module.__init__(self)
        super(Net, self).__init__()
        self.bottom = sharedBottom()
        self.sem_seg = laneNet()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, y): 
        x = self.bottom(x, y)
        x_sem = self.sem_seg(x)
        return x_sem

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.x64 = nn.Conv2d(64,1,kernel_size=1, padding = 0, bias = False)
        self.x128 = nn.Conv2d(128,1,kernel_size=1, padding = 0, bias = False)
        self.x256 = nn.Conv2d(256,1,kernel_size=1, padding = 0, bias = False)
        self.x64.weight = torch.nn.Parameter(torch.ones((64,1,1,1)))
        self.x128.weight = torch.nn.Parameter(torch.ones((128,1,1,1)))
        self.x256.weight = torch.nn.Parameter(torch.ones((256,1,1,1)))
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding = 1, bias=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding = 1, bias=True)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding = 1, bias=True)
        params = torch.load('/mnt/lustre/dingmingyu/.torch/models/vgg19-dcbb9e9d.pth')
        self.conv1_2.weight = torch.nn.Parameter(params['features.2.weight'])
        self.conv1_2.bias = torch.nn.Parameter(params['features.2.bias'])
        self.conv2_2.weight = torch.nn.Parameter(params['features.7.weight'])
        self.conv2_2.bias = torch.nn.Parameter(params['features.7.bias'])
        self.conv3_4.weight = torch.nn.Parameter(params['features.16.weight'])
        self.conv3_4.bias = torch.nn.Parameter(params['features.16.bias'])
        #{k: v for k, v in pretrained_dict.items() if k in model_dict}        
    def forward(self, x): 
        x64 = self.x64(x)
        x64 = self.conv1_2(x64)
        x128 = self.x128(x)
        x128 = self.conv2_2(x128)
        x256 = self.x256(x)
        x256 = self.conv3_4(x256)
        x_vgg = torch.cat([x64, x128, x256], dim = 1)
        return x_vgg