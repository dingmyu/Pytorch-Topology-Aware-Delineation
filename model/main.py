import shutil
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torchvision import transforms
from torch.utils.data import DataLoader
from loss import *
from dataloader import *
import lanenet
import os
import torch.nn.functional as F
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
parser = argparse.ArgumentParser()
parser.add_argument('--dir_path', default='')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=16, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--new_length', default=705, type=int)
parser.add_argument('--new_width', default=833, type=int)
parser.add_argument('--label_length', default=177, type=int)
parser.add_argument('--label_width', default=209, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--save-freq', default=1, type=int,
                    metavar='N', help='save frequency (default: 200)')
parser.add_argument('--resume', default='checkpoints', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

best_prec = 0


def main():
    global args, best_prec
    args = parser.parse_args()
    print ("Build model ...")
    model = lanenet.Net()
    model = torch.nn.DataParallel(model).cuda()
    #model.apply(weights_init) 
    #params = torch.load('checkpoints/001_single_module.pth.tar')  # finetune from single channel
    #bs, n , h, w = params['state_dict']['module.bottom.conv1.cbr_unit.0.weight'].size()
    #params['state_dict']['module.bottom.conv1.cbr_unit.0.weight'] = torch.cat((params['state_dict']['module.bottom.conv1.cbr_unit.0.weight'],torch.randn((bs,1,h,w)).float().cuda()),1)
    #model.load_state_dict(params['state_dict'])

    params = torch.load('checkpoints/001_checkpoint.pth.tar') 
    model.load_state_dict(params['state_dict'])
    args.start_epoch= params['epoch']
    model_vgg = lanenet.VGG()
    model_vgg = torch.nn.DataParallel(model_vgg).cuda()
    if not os.path.exists(args.resume):
        os.makedirs(args.resume)
    print("Saving everything to directory %s." % (args.resume))

    # define loss function (criterion) and optimizer
    criterion = cross_entropy2d
    #criterion_mse = torch.nn.DataParallel(torch.nn.MSELoss()).cuda()  #this DataParallel is useless, there will still be gather but not parallel.
    criterion_mse = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    optimizer.load_state_dict(params['optimizer'])
    
    #optimizer = nn.DataParallel(optimizer)   # no use
    cudnn.benchmark = True

    # data transform
    
    train_data = MyDataset('/mnt/lustre/share/dingmingyu/new_list_lane.txt', args.dir_path, args.new_width, args.new_length,args.label_width,args.label_length)

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,pin_memory=True)


    for epoch in range(args.start_epoch, args.epochs):
        print 'epoch: ' + str(epoch + 1)

        # train for one epoch
        train(train_loader, model, model_vgg, criterion, criterion_mse, optimizer, epoch)

        # evaluate on validation set

        # remember best prec and save checkpoint

        if (epoch + 1) % args.save_freq == 0:
            checkpoint_name = "%03d_%s" % (epoch + 1, "checkpoint.pth.tar")
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, checkpoint_name, args.resume)



def train(train_loader, model, model_vgg, criterion, criterion_mse, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_seg = AverageMeter()
    losses_seg1 = AverageMeter()
    losses_seg2 = AverageMeter()
    losses_vgg = AverageMeter()
    losses_vgg1 = AverageMeter()
    losses_vgg2 = AverageMeter()
    lrs = AverageMeter()
    # switch to train mode
    model.train()
    model_vgg.train()
    weight_cus = torch.ones(2)
    weight_cus[1] = 2
    weight_cus = weight_cus.cuda()
    end = time.time()
    for i, (input, target_ins, n_objects) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        lr = adjust_learning_rate(optimizer, epoch*len(train_loader)+i, args.epochs*len(train_loader))
        lrs.update(lr)
        device = torch.cuda.current_device() #useless
        input = input.float().cuda(device)
        target01 = target_ins.sum(1)
        target01 = target01.long().cuda(device)

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target01)
        
        bs, n , h, w = input.size()
        result_var = torch.autograd.Variable(torch.zeros((bs, 1, h, w)).float().cuda(device))

        x_sem = model(input_var, result_var)    
        x_prob = nn.functional.softmax(x_sem, dim=1)[:,1,:,:]
        x_vgg = model_vgg(x_prob.unsqueeze(1))
        y_vgg = model_vgg(target_var.unsqueeze(1).float())
        y_vgg_nograd = y_vgg.detach()
        loss_vgg = criterion_mse(x_vgg, y_vgg_nograd)
        
        loss_seg = criterion(x_sem, target_var, size_average=True)
        x_sem = torch.nn.functional.softmax(x_sem,dim=1)[:,1,:,:].unsqueeze(1).contiguous()
        x_sem1 = model(input_var, x_sem)
        
        x_prob1 = nn.functional.softmax(x_sem1, dim=1)[:,1,:,:]
        x_vgg1 = model_vgg(x_prob1.unsqueeze(1))
        
        loss_vgg1 = criterion_mse(x_vgg1, y_vgg_nograd)
        loss_seg1 = criterion(x_sem1, target_var, size_average=True)
        x_sem1 = torch.nn.functional.softmax(x_sem1,dim=1)[:,1,:,:].unsqueeze(1).contiguous() 
        x_sem2 = model(input_var, x_sem1)
        x_prob2 = nn.functional.softmax(x_sem2, dim=1)[:,1,:,:]
        x_vgg2 = model_vgg(x_prob2.unsqueeze(1))  
        loss_vgg2 = criterion_mse(x_vgg2, y_vgg_nograd)
        
        loss_seg2 = criterion(x_sem2, target_var, size_average=True)
        loss = loss_seg + 2 * loss_seg1 + 3 * loss_seg2 + 0.1 * (loss_vgg+ 2*loss_vgg1 + 3*loss_vgg2)

        losses_seg.update(loss_seg.data[0], input.size(0))
        losses_seg1.update(loss_seg1.data[0], input.size(0))
        losses_seg2.update(loss_seg2.data[0], input.size(0))
        losses_vgg.update(loss_vgg.data[0], input.size(0))
        losses_vgg1.update(loss_vgg1.data[0], input.size(0))
        losses_vgg2.update(loss_vgg2.data[0], input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()  # there is no need sum
        #optimizer.module.step()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss_seg {loss_seg.val:.4f} ({loss_seg.avg:.4f})\t'
                  'Loss_seg1 {loss_seg1.val:.4f} ({loss_seg1.avg:.4f})\t'
                  'Loss_seg2 {loss_seg2.val:.4f} ({loss_seg2.avg:.4f})\t'
                  'Loss_vgg {loss_vgg.val:.4f} ({loss_vgg.avg:.4f})\t'
                  'Loss_vgg1 {loss_vgg1.val:.4f} ({loss_vgg1.avg:.4f})\t'
                  'Loss_vgg2 {loss_vgg2.val:.4f} ({loss_vgg2.avg:.4f})\t'
                  'Lr {lr.val:.5f} ({lr.avg:.5f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss_seg=losses_seg, loss_seg1=losses_seg1, loss_seg2=losses_seg2, loss_vgg=losses_vgg, loss_vgg1=losses_vgg1, loss_vgg2=losses_vgg2, lr=lrs))



def save_checkpoint(state, filename, resume_path):
    cur_path = os.path.join(resume_path, filename)
    torch.save(state, cur_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, curr_iter, max_iter, power=0.9):
    lr = args.lr * (1 - float(curr_iter)/max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

if __name__ == '__main__':
    main()
