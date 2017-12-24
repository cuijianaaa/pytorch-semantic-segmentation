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

ckpt_path = '../../ckpt'
exp_name = 'cityscapes-psp_net'
writer = SummaryWriter(os.path.join(ckpt_path, 'exp', exp_name))

args = {
    'train_batch_size': 1,
    'val_batch_size': 1,
    'lr': 1e-2 / sqrt(16 / 2),
    'lr_decay': 0.9,
    'max_iter': 9e4,
    'longer_size': 2048,
    'crop_size': 712,
    'stride_rate': 2 / 3.,
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
    net = (lambda: InsNet())()
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

    train_set = cityscapes.Instance('train', joint_transform=train_joint_transform, sliding_crop=sliding_crop, transform=train_input_transform, target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=8, shuffle=True)
    val_set = cityscapes.Instance('val', transform=val_input_transform, sliding_crop=sliding_crop, target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=8, shuffle=False)

    criterion = CrossEntropyLoss2d(size_average=True, ignore_index=cityscapes.ignore_label).cuda()

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

    train(train_loader, net, criterion, optimizer, curr_epoch, args, val_loader, visualize)


def train(train_loader, net, criterion, optimizer, curr_epoch, train_args, val_loader, visualize):
    while True:
        train_main_loss = AverageMeter()
        train_aux_loss = AverageMeter()
        curr_iter = (curr_epoch - 1) * len(train_loader)
        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * train_args['lr'] * (1 - float(curr_iter) / train_args['max_iter']
                                                                      ) ** train_args['lr_decay']
            optimizer.param_groups[1]['lr'] = train_args['lr'] * (1 - float(curr_iter) / train_args['max_iter']
                                                                  ) ** train_args['lr_decay']

            inputs, gts, ins, _ = data
            #print gts
            #print ins
            assert len(inputs.size()) == 5 and len(gts.size()) == 4
            inputs.transpose_(0, 1)
            gts.transpose_(0, 1)

            assert inputs.size()[3:] == gts.size()[2:]
            slice_batch_pixel_size = inputs.size(1) * inputs.size(3) * inputs.size(4)

            for inputs_slice, gts_slice in zip(inputs, gts):
                inputs_slice = Variable(inputs_slice).cuda()
                gts_slice = Variable(gts_slice).cuda()

                optimizer.zero_grad()
                outputs, ins_code, aux = net(inputs_slice)
                print ins_code
                print ins_code.size()            
                #print 'outputs ',outputs.size()
                #print 'aux ', aux.size()
                #print 'gt ',gts_slice.size()
                #print outputs.size()
                #print gts_slice.size()
                #print cityscapes.num_classes
                assert outputs.size()[2:] == gts_slice.size()[1:]
                assert outputs.size()[1] == cityscapes.num_classes

                main_loss = criterion(outputs, gts_slice)
                aux_loss = criterion(aux, gts_slice)
                loss = main_loss + 0.4 * aux_loss
                loss.backward()
                optimizer.step()

                train_main_loss.update(main_loss.data[0], slice_batch_pixel_size)
                train_aux_loss.update(aux_loss.data[0], slice_batch_pixel_size)

            curr_iter += 1
            writer.add_scalar('train_main_loss', train_main_loss.avg, curr_iter)
            writer.add_scalar('train_aux_loss', train_aux_loss.avg, curr_iter)
            writer.add_scalar('lr', optimizer.param_groups[1]['lr'], curr_iter)

            if (i + 1) % train_args['print_freq'] == 0:
                print('[epoch %d], [iter %d / %d], [train main loss %.5f], [train aux loss %.5f]. [lr %.10f]' % (
                    curr_epoch, i + 1, len(train_loader), train_main_loss.avg, train_aux_loss.avg,
                    optimizer.param_groups[1]['lr']))
            #if (i + 1) % train_args['print_freq'] == 0:
            #    print('[epoch %d], [iter %d / %d], [train main loss %.5f], [lr %.10f]' % (
            #        curr_epoch, i + 1, len(train_loader), train_main_loss.avg, #train_aux_loss.avg,
            #        optimizer.param_groups[1]['lr']))

            if curr_iter >= train_args['max_iter']:
                return
            #if curr_iter % train_args['val_freq'] == 0:
            #validate(val_loader, net, criterion, optimizer, curr_epoch, i + 1, train_args, visualize)
        validate(val_loader, net, criterion, optimizer, curr_epoch, 0, train_args, visualize)
        curr_epoch += 1


def validate(val_loader, net, criterion, optimizer, epoch, iter_num, train_args, visualize):
    # the following code is written assuming that batch size is 1
    net.eval()

    val_loss = AverageMeter()

    gts_all = np.zeros((len(val_loader), args['longer_size'] / 2, args['longer_size']), dtype=int)
    predictions_all = np.zeros((len(val_loader), args['longer_size'] / 2, args['longer_size']), dtype=int)
    for vi, data in enumerate(val_loader):
        input, gt, ins, slices_info = data
        assert len(input.size()) == 5 and len(gt.size()) == 4 and len(slices_info.size()) == 3
        input.transpose_(0, 1)
        gt.transpose_(0, 1)
        slices_info.squeeze_(0)
        assert input.size()[3:] == gt.size()[2:]

        count = torch.zeros(args['longer_size'] / 2, args['longer_size']).cuda()
        output = torch.zeros(cityscapes.num_classes, args['longer_size'] / 2, args['longer_size']).cuda()

        slice_batch_pixel_size = input.size(1) * input.size(3) * input.size(4)

        for input_slice, gt_slice, info in zip(input, gt, slices_info):
            input_slice = Variable(input_slice).cuda()
            gt_slice = Variable(gt_slice).cuda()

            output_slice, ins_slice,  _  = net(input_slice)
            assert output_slice.size()[2:] == gt_slice.size()[1:]
            assert output_slice.size()[1] == cityscapes.num_classes
            if(len(info.size())<2):
                info = info.view(1,6)

            for i in range(info.size()[0]):
                output[:, info[i,0]: info[i,1], info[i,2]: info[i,3]] += output_slice[0, :, :info[i,4], :info[i,5]].data
                output[:, info[i,0]: info[i,1], info[i,2]: info[i,3]] += output_slice[0, :, :info[i,4], :info[i,5]].data
                gts_all[vi, info[i,0]: info[i,1], info[i,2]: info[i,3]] += gt_slice[0, :info[i,4], :info[i,5]].data.cpu().numpy()
                count[info[i,0]: info[i,1], info[i,2]: info[i,3]] += 1

            val_loss.update(criterion(output_slice, gt_slice).data[0], slice_batch_pixel_size)

        output /= count
        gts_all[vi, :, :] /= count.cpu().numpy().astype(int)
        predictions_all[vi, :, :] = output.max(0)[1].squeeze_(0).cpu().numpy()

        print('validating: %d / %d' % (vi + 1, len(val_loader)))
    acc, acc_cls, mean_iu, fwavacc = evaluate(predictions_all, gts_all, cityscapes.num_classes)
    if val_loss.avg < train_args['best_record']['val_loss']:
        train_args['best_record']['val_loss'] = val_loss.avg
        train_args['best_record']['epoch'] = epoch
        train_args['best_record']['iter'] = iter_num
        train_args['best_record']['acc'] = acc
        train_args['best_record']['acc_cls'] = acc_cls
        train_args['best_record']['mean_iu'] = mean_iu
        train_args['best_record']['fwavacc'] = fwavacc
    snapshot_name = 'epoch_%d_iter_%d_loss_%.5f_acc_%.5f_acc-cls_%.5f_mean-iu_%.5f_fwavacc_%.5f_lr_%.10f' % (
        epoch, iter_num, val_loss.avg, acc, acc_cls, mean_iu, fwavacc, optimizer.param_groups[1]['lr'])
    torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '.pth'))
    torch.save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name, 'opt_' + snapshot_name + '.pth'))

    if train_args['val_save_to_img_file']:
        to_save_dir = os.path.join(ckpt_path, exp_name, '%d_%d' % (epoch, iter_num))
        check_mkdir(to_save_dir)

    val_visual = []
    for idx, data in enumerate(zip(gts_all, predictions_all)):
        gt_pil = cityscapes.colorize_mask(data[0])
        predictions_pil = cityscapes.colorize_mask(data[1])
        if train_args['val_save_to_img_file']:
            predictions_pil.save(os.path.join(to_save_dir, '%d_prediction.png' % idx))
            gt_pil.save(os.path.join(to_save_dir, '%d_gt.png' % idx))
            val_visual.extend([visualize(gt_pil.convert('RGB')),
                               visualize(predictions_pil.convert('RGB'))])
    if(not (len(val_visual) == 0)):
        val_visual = torch.stack(val_visual, 0)
        val_visual = vutils.make_grid(val_visual, nrow=2, padding=5)
        writer.add_image(snapshot_name, val_visual)

    print('-----------------------------------------------------------------------------------------------------------')
    print('[epoch %d], [iter %d], [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f]' % (
        epoch, iter_num, val_loss.avg, acc, acc_cls, mean_iu, fwavacc))

    print('best record: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [epoch %d], '
          '[iter %d]' % (train_args['best_record']['val_loss'], train_args['best_record']['acc'],
                         train_args['best_record']['acc_cls'], train_args['best_record']['mean_iu'],
                         train_args['best_record']['fwavacc'], train_args['best_record']['epoch'],
                         train_args['best_record']['iter']))

    print('-----------------------------------------------------------------------------------------------------------')

    writer.add_scalar('val_loss', val_loss.avg, epoch)
    writer.add_scalar('acc', acc, epoch)
    writer.add_scalar('acc_cls', acc_cls, epoch)
    writer.add_scalar('mean_iu', mean_iu, epoch)
    writer.add_scalar('fwavacc', fwavacc, epoch)

    net.train()
    return val_loss.avg


if __name__ == '__main__':
    main()
