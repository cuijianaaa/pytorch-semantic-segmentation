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
from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d
from loss import InstanceLoss
import matplotlib.pyplot as plt
from skimage import color
import cv2

ckpt_path = '../../ckpt'
#exp_name = 'cityscapes-psp_net'
exp_name = 'ins_train'
writer = SummaryWriter(os.path.join(ckpt_path, 'exp', exp_name))

args = {
    'train_batch_size': 1,
    'val_batch_size': 1,
    'lr': 0 * 1e-2 / sqrt(16 / 2),
    'lr_decay': 0.9,
    'max_iter': 9e4,
    'longer_size': 2048,
    'crop_size': 712,
    'stride_rate': 2 / 3.,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'snapshot': 'epoch_98_loss_4.82587_ins_0.04282.pth',
    'print_freq': 10,
    'val_save_to_img_file': True,
    'val_img_sample_rate': 0.01,  # randomly sample some validation results to display,
    'val_img_display_size': 384,
    'val_freq': 400
}


def main():
    #net = PSPNet(num_classes=cityscapes.num_classes)
    #net = (lambda: PSPNetInstance(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'))()
    net = (lambda: InsNet())()
    if len(args['snapshot']) != 0:
        print('training resumes from ' + args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'])))
    curr_epoch = 0
    net.cuda().train()

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_joint_transform = joint_transforms.ComposeInstance([
        joint_transforms.ScaleInstance(args['longer_size']),
        joint_transforms.RandomRotateInstance(10),
        joint_transforms.RandomHorizontallyFlipInstance()
    ])
    sliding_crop = joint_transforms.SlidingCropInstance(args['crop_size'], args['stride_rate'], cityscapes.ignore_label)
    train_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    val_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    target_transform = extended_transforms.MaskToTensor()
    visualize = standard_transforms.Compose([
        standard_transforms.Scale(args['val_img_display_size']),
        standard_transforms.ToTensor()
    ])

    #train_set = cityscapes.Instance('train', joint_transform=train_joint_transform, sliding_crop=sliding_crop, transform=train_input_transform, target_transform=target_transform)
    train_set = cityscapes.Instance('train', joint_transform=None, sliding_crop=sliding_crop, transform=train_input_transform, target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=8, shuffle=True)
    val_set = cityscapes.Instance('val', transform=val_input_transform, sliding_crop=sliding_crop, target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=8, shuffle=False)

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
    #validate(train_loader, net, criterion, criterion_ins, optimizer, curr_epoch, 0, args, visualize)
    train(train_loader, net, criterion, criterion_ins, optimizer, curr_epoch, args, val_loader, visualize)


def train(train_loader, net, criterion, criterion_ins, optimizer, curr_epoch, train_args, val_loader, visualize):
    nnn = 0
    min_loss = 100;
    while True:
        train_main_loss = AverageMeter()
        train_aux_loss = AverageMeter()
        train_ins_loss = AverageMeter()
        curr_iter = (curr_epoch - 1) * len(train_loader)
        gts_all = np.zeros((len(train_loader), args['longer_size'] / 2, args['longer_size']), dtype=int)
        gts_all_list = []
        predictions_all = np.zeros((len(train_loader), args['longer_size'] / 2, args['longer_size']), dtype=int)
        predictions_all_ins = []#np.zeros((len(train_loader), args['longer_size'] / 2, args['longer_size']), dtype=int)
        for vi, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * train_args['lr'] * (1 - float(curr_iter) / train_args['max_iter']
                                                                      ) ** train_args['lr_decay']
            optimizer.param_groups[1]['lr'] = train_args['lr'] * (1 - float(curr_iter) / train_args['max_iter']
                                                                       ) ** train_args['lr_decay']
            #inputs, gts, ins, _ = data
            inputs, gts, ins, slices_info = data
            print nnn
            nnn = nnn + 1
            assert len(inputs.size()) == 5 and len(gts.size()) == 4
            inputs.transpose_(0, 1)
            gts.transpose_(0, 1)
            ins.transpose_(0, 1)
            assert len(inputs.size()) == 5 and len(gts.size()) == 4 and len(slices_info.size()) == 3
            assert inputs.size()[3:] == gts.size()[2:]
            slices_info.squeeze_(0)
            slice_batch_pixel_size = inputs.size(1) * inputs.size(3) * inputs.size(4)
            #for inputs_slice, info in zip(inputs, slices_info):
            #for inputs_slice in inputs:
            count = torch.zeros(args['longer_size'] / 2, args['longer_size'])
            output = torch.zeros(cityscapes.num_classes, args['longer_size'] / 2, args['longer_size'])
            output_ins = [] 
            gts_list = []
            #print 'slices_info.size ', slices_info.size()
            for inputs_slice, gts_slice, ins_slice, info in zip(inputs, gts, ins, slices_info):
            #for inputs_slice, gts_slice, ins_slice in zip(inputs, gts, ins):
                inputs_slice = Variable(inputs_slice).cuda()
                gts_slice = Variable(gts_slice).cuda()
                ins_slice = Variable(ins_slice).cuda()
                optimizer.zero_grad()
                output_slice, ins_code, aux = net(inputs_slice)
                print 'output_slice size ', output_slice.size()
                print 'ins_slice size ', ins_slice.size()
                assert output_slice.size()[2:] == gts_slice.size()[1:]
                print 'ins_code size', ins_code.size()
                assert output_slice.size()[1] == cityscapes.num_classes
                #print info
                #print 'info size ', info.size()
                #if(len(info.size())<2):
                #    info = info.view(1,6)
                #plt.figure(i)
                ##plt.imshow(gts_slice[0, :info[i,4], :info[i,5]].data.cpu().numpy())
                ins_code_numpy = ins_code[0, :, :info[4], :info[5]].data.cpu().numpy()
                ins_code_numpy[ins_code_numpy > 0.5] = 1
                ins_code_numpy[ins_code_numpy < 1] = 0
                ins_code_numpy = ins_code_numpy.astype(int)
                ins_code_gry = np.zeros((info[4], info[5]), dtype=int)
                for bit in range(8):
                    ins_code_gry += ins_code_numpy[bit,:,:] * (2 ** bit)
                ins_code_gry = ins_code_gry.astype(np.uint8)

                output_ins.append(ins_code_gry)
                gts_list.append(gts_slice[0, :info[4], :info[5]].data.cpu().numpy())
                output[:, info[0]: info[1], info[2]: info[3]] += output_slice[0, :, :info[4], :info[5]].data.cpu()
                gts_all[vi, info[0]: info[1], info[2]: info[3]] += gts_slice[0, :info[4], :info[5]].data.cpu().numpy()
                count[info[0]: info[1], info[2]: info[3]] += 1
                main_loss = criterion(output_slice, gts_slice)
                aux_loss = criterion(aux, gts_slice)
                ins_loss = 100 * criterion_ins(ins_code, ins_slice.unsqueeze(dim = 1), gts_slice.unsqueeze(dim = 1))
                loss = main_loss + 0.4 * aux_loss + ins_loss
                #print 'main loss', main_loss
                #print 'ins_loss', ins_loss
                loss.backward()
                #optimizer.step()
            output /= count
            gts_all[vi, :, :] /= count.cpu().numpy().astype(int)
            #print gts_all[vi,:,:]
            #for i in range(256):
            #    n = np.sum(gts_all[vi,:,:] == i)
            #    if n > 0:
            #        print '%d number: %d'%(i,n)
            predictions_all[vi, :, :] = output.max(0)[1].squeeze_(0).cpu().numpy()

            predictions_all_ins.append(output_ins)
            gts_all_list.append(gts_list)
            if train_args['val_save_to_img_file']:
                to_save_dir = os.path.join(ckpt_path, exp_name, 'img')
                check_mkdir(to_save_dir)

            for idx, data in enumerate(zip(gts_all, predictions_all)):
                gt_pil = cityscapes.colorize_mask(data[0])
                predictions_pil = cityscapes.colorize_mask(data[1])
                codes = predictions_all_ins[idx]
                gtss = gts_all_list[idx]
                print 'codes len ', len(codes)
                if train_args['val_save_to_img_file']:
                    predictions_pil.save(os.path.join(to_save_dir, '%d_prediction.png' % idx))
                    gt_pil.save(os.path.join(to_save_dir, '%d_gt.png' % idx))
                    for jj in range(len(codes)):
                        code = codes[jj]
                        code_rgb = color.label2rgb(code) * 255
                        gts = gtss[jj]
                        gts_rgb = color.label2rgb(gts) * 255
                        print 'ins gry shape ',code.shape
                        print 'ins type ', code.dtype
                        print '\n'
                        print 'ins rgb gry shape ',code_rgb.shape
                        print 'ins rgb type ', code_rgb.dtype
                        print np.max(code_rgb),' ',np.min(code_rgb)
                        print '\n'
                        print 'gts rgb shape ',gts_rgb.shape
                        print 'gts rgb type ', gts_rgb.dtype
                        print '\n'
                        cv2.imwrite(os.path.join(to_save_dir, '%d_%d_ins_gry.png' % (idx, jj)), code.astype(np.uint8))
                        cv2.imwrite(os.path.join(to_save_dir, '%d_%d_ins_rgb.png' % (idx, jj)), code_rgb.astype(np.uint8))
                        cv2.imwrite(os.path.join(to_save_dir, '%d_%d_gts_rgb.png' % (idx, jj)), gts_rgb.astype(np.uint8))

        break



        


        


def validate(val_loader, net, criterion, criterion_ins, optimizer, epoch, iter_num, train_args, visualize):

    predictions_all = np.zeros((len(val_loader), args['longer_size'] / 2, args['longer_size']), dtype=int)
    for vi, data in enumerate(val_loader):
        input, gt, ins, slices_info = data
        assert len(input.size()) == 5 and len(gt.size()) == 4 and len(slices_info.size()) == 3
        input.transpose_(0, 1)
        gt.transpose_(0, 1)
        ins.transpose_(0, 1)
        slices_info.squeeze_(0)
        assert input.size()[3:] == gt.size()[2:]
        print 'test2'
        count = torch.zeros(args['longer_size'] / 2, args['longer_size'])
        output = torch.zeros(cityscapes.num_classes, args['longer_size'] / 2, args['longer_size'])

        for input_slice, gt_slice,  info in zip(input, gt, slices_info):
            output_slice, ins_code,  _  = net(input_slice)
            if(len(info.size())<2):
                info = info.view(1,6)
            for i in range(info.size()[0]):
                output[:, info[i,0]: info[i,1], info[i,2]: info[i,3]] += output_slice[0, :, :info[i,4], :info[i,5]].data.cpu()
                count[info[i,0]: info[i,1], info[i,2]: info[i,3]] += 1
        output /= count
        gts_all[vi, :, :] /= count.cpu().numpy().astype(int)
        predictions_all[vi, :, :] = output.max(0)[1].squeeze_(0).cpu().numpy()

               
        print('validating: %d / %d' % (vi + 1, len(val_loader)))
    acc, acc_cls, mean_iu, fwavacc = evaluate(predictions_all, gts_all, cityscapes.num_classes)
    
    if train_args['val_save_to_img_file']:
        to_save_dir = os.path.join(ckpt_path, exp_name, '%d_%d' % (epoch, iter_num))
        check_mkdir(to_save_dir)

    for idx, data in enumerate(zip(gts_all, predictions_all)):
        gt_pil = cityscapes.colorize_mask(data[0])
        predictions_pil = cityscapes.colorize_mask(data[1])
        if train_args['val_save_to_img_file']:
            predictions_pil.save(os.path.join(to_save_dir, '%d_prediction.png' % idx))
            gt_pil.save(os.path.join(to_save_dir, '%d_gt.png' % idx))



if __name__ == '__main__':
    main()
