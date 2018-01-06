import torch
from torch import nn
from torch.nn import functional as F

import extractors
from utils import initialize_weights                                                                           
import os

class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PSPUpsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


class PSPNet1(nn.Module):
    def __init__(self, n_classes=19, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet34',
                 pretrained=True):
        super(PSPNet1, self).__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.LogSoftmax()
        )

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

        self.aux_logits = nn.Conv2d(256, n_classes, kernel_size=1)
        initialize_weights(self.aux_logits)


    def forward(self, x):
        x_size = x.size()
        f, class_f = self.feats(x) 
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)
#        print 'class_f', class_f.size()
        #auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))

        #return self.final(p), self.classifier(auxiliary)
        aux = self.aux_logits(class_f)

        return self.final(p), F.upsample(aux, x_size[2:], mode='bilinear')


class PSPNetInstance(nn.Module):
    def __init__(self, n_classes=19, ins_bit = 8, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet34',
                 pretrained=True):
        super(PSPNetInstance, self).__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.LogSoftmax()
        )
        #self.ins = nn.Sequential(
        #    nn.Conv2d(64, ins_bit, kernel_size=1),
        #    nn.Sigmoid()
        #)

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )
        self.aux_logits = nn.Conv2d(256, n_classes, kernel_size=1)
        initialize_weights(self.aux_logits)


    def forward(self, x):
        x_size = x.size()
        f, class_f = self.feats(x) 
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)

        aux = self.aux_logits(class_f)

        return self.final(p), p, F.upsample(aux, x_size[2:], mode='bilinear')


class InsNet(nn.Module):
    def __init__(self, ins_bit = 8, pretrained=True):
        super(InsNet, self).__init__()
        self.psp = (lambda: PSPNetInstance(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'))() 
        self.psp.load_state_dict(torch.load(os.path.join('/home/cj/pytorch/pytorch-parsing/ckpt/cityscapes-psp_net/epoch_71_iter_0_loss_0.21447_acc_0.93891_acc-cls_0.72612_mean-iu_0.63503_fwavacc_0.89032_lr_0.0015954517.pth')))
        self.ins = nn.Sequential(
            nn.Conv2d(64, ins_bit, kernel_size=1),
            nn.Sigmoid()
        )
        initialize_weights(self.ins)


    def forward(self, x):
        final_layer_output, pre_layer_output, aux = self.psp(x) 
        ins_code = self.ins(pre_layer_output)  
        return final_layer_output, ins_code, aux 
