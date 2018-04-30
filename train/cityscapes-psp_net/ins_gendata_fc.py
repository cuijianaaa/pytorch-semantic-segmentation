import datetime
import os
from math import sqrt
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import torchvision.transforms as standard_transforms 
from tensorboardX import SummaryWriter
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import utils.joint_transforms as joint_transforms
import utils.transforms as extended_transforms
from datasets import cityscapes
from models import *
from models.pspnet import PSPNet1
from models.pspnet import PSPNetInstance
from models.pspnet import InsNet
from models.pspnet import FCNet
from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d
from loss import InstanceLoss

from skimage import color
import cv2
import matplotlib.pyplot as plt
ckpt_path = '../../ckpt'
exp_name = 'ins_train'
writer = SummaryWriter(os.path.join(ckpt_path, 'exp', exp_name))

args = {
    'train_batch_size': 1,
    'val_batch_size': 1,
    'lr': 100,
    'lr_decay': 0.9,
    'max_iter': 9e4,
    'longer_size': 2048,
    'crop_size': 712,
    #'stride_rate': 2 / 3.,
    'stride_rate': 1.,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'snapshot': '',
    'print_freq': 10,
    'val_save_to_img_file': False,
    'val_img_sample_rate': 0.01,  # randomly sample some validation results to display,
    'val_img_display_size': 384,
    'val_freq': 400
}


def main():
    #net = PSPNet(num_classes=cityscapes.num_classes)
    #net = (lambda: PSPNetInstance(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'))()
    #net = (lambda: InsNet())()
    net = (lambda: FCNet())()
    if len(args['snapshot']) == 0:
        # net.load_state_dict(torch.load(os.path.join(ckpt_path, 'cityscapes (coarse)-psp_net', 'xx.pth')))
        curr_epoch = 1
        args['best_record'] = {'epoch': 0, 'iter': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0,
                               'fwavacc': 0}
    else:
        print('training resumes from ' + args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'])))
        split_snapshot = args['snapshot'].split('_')
        curr_epoch = int(split_snapshot[1]) + 1
        args['best_record'] = {'epoch': int(split_snapshot[1]), 'iter': int(split_snapshot[3]),
                               'val_loss': float(split_snapshot[5]), 'acc': float(split_snapshot[7]),
                               'acc_cls': float(split_snapshot[9]),'mean_iu': float(split_snapshot[11]),
                               'fwavacc': float(split_snapshot[13])}
    net.cuda().train()


#    train_joint_transform = joint_transforms.ComposeInstance([
#        joint_transforms.ResizeInstance(712),
#    ])

    train_joint_transform = joint_transforms.ComposeInstance([
        joint_transforms.CenterCropInstance(712),
        #joint_transforms.ResizeInstance(512),
    ])
    train_set = cityscapes.InstanceGenData(
        'train', 
        joint_transform=None,
        transform=standard_transforms.ToTensor(),
        target_transform=extended_transforms.MaskToTensor())

    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=8, shuffle=False)

    criterion = CrossEntropyLoss2d(size_average=True, ignore_index=cityscapes.ignore_label).cuda()
    criterion_ins = InstanceLoss(class_num = 19, code_bit = 8).cuda()
    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'], nesterov=True)

    if len(args['snapshot']) > 0:
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, 'opt_' + args['snapshot'])))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt'), 'w').write(str(args) + '\n\n')

    train(train_loader, net, criterion, criterion_ins, optimizer, curr_epoch, args)


def train(train_loader, net, criterion, criterion_ins, optimizer, curr_epoch, train_args):
    nnn = 0
    min_loss = 100;
    while True:
        train_main_loss = AverageMeter()
        train_aux_loss = AverageMeter()
        train_ins_loss = AverageMeter()
        curr_iter = (curr_epoch - 1) * len(train_loader)
        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * train_args['lr'] * (1 - float(curr_iter) / train_args['max_iter']
                                                                      ) ** train_args['lr_decay']
            optimizer.param_groups[1]['lr'] = train_args['lr'] * (1 - float(curr_iter) / train_args['max_iter']
                                                                  ) ** train_args['lr_decay']

            inputs, gts, ins = data
            print("input size ", inputs.size())
            print("gts size ", gts.size())
            print("ins size ", ins.size())
            print nnn
            nnn = nnn + 1
            assert len(inputs.size()) == 4 and len(gts.size()) == 3
            #inputs.transpose_(0, 1)
            #gts.transpose_(0, 1)
            #ins.transpose_(0, 1)
            assert inputs.size()[2:] == gts.size()[1:]
            slice_batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)
            inputs = Variable(inputs).cuda()
            gts = Variable(gts).cuda()
            ins = Variable(ins).cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            print("outputs size ", outputs.size()) 

            assert outputs.size()[2:] == gts.size()[1:]
            assert outputs.size()[1] == cityscapes.num_classes
            src_rgb = (inputs[0,:,:,:].data.cpu().numpy().transpose([1,2,0]) + 128) * 128

            class_gt_rgb = color.label2rgb(gts[0,:,:].data.cpu().numpy()) * 255
            class_gt_rgb = class_gt_rgb.astype(np.uint8)
            class_max, class_id = torch.max(outputs[0,:,:,:].data, dim=0) 
            class_rgb = color.label2rgb(class_id.cpu().numpy()) * 255
            class_rgb = class_rgb.astype(np.uint8) 
            plt.figure(1)
            plt.imshow((src_rgb).astype(np.uint8))
            plt.figure(2)
            plt.imshow((class_gt_rgb * 0.5 + src_rgb * 0.5).astype(np.uint8))
            plt.figure(3)
            plt.imshow((class_rgb * 0.5 + src_rgb * 0.5).astype(np.uint8))
            main_loss = criterion(outputs, gts)
            loss = main_loss
            #loss = ins_loss
            loss.backward()
            optimizer.step()

            train_main_loss.update(main_loss.data[0], slice_batch_pixel_size)
            plt.pause(1)
            curr_iter += 1
            writer.add_scalar('train_main_loss', train_main_loss.avg, curr_iter)
            writer.add_scalar('lr', optimizer.param_groups[1]['lr'], curr_iter)

            print('[epoch %d], [iter %d / %d], [train main loss %.5f], [lr %.10f]' % (
                    curr_epoch, i + 1, len(train_loader), train_main_loss.avg,
                    optimizer.param_groups[1]['lr']))





            if curr_iter >= train_args['max_iter']:
                return



        if train_ins_loss.avg < min_loss:
            min_loss = train_ins_loss.avg
            snapshot_name = 'epoch_%d_loss_%.5f_ins_%.5f' % (
                curr_epoch, train_main_loss.avg, train_ins_loss.avg)
            #torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '.pth'))
            #torch.save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name, 'opt_' + snapshot_name + '.pth'))

        curr_epoch += 1



if __name__ == '__main__':
    main()
