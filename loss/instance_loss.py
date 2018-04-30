#net.out.class_num = w * h * 19
#net.out.instance_num = w * h * 8
#gt.class = w * h * 1
#gt.instance = w * h * 1
#Loss_in = 0
#Loss_out = 0
#for c in all class(0-18)
#    pixels[c] = in gt find c
#    max_obj_num = from json   
#    for obj1 in range(0, max_obj_num)
#        find pixels[c][obj1] = in pixels[c] find obj
#        mean[c][obj1] = mean(net.out.instance[pixels[c][obj1]]) # net.out.instance[pixels]  nx8  mean 1x8
#        tmp = (net.out.instance[pixels[c][obj1]] - mean[c][obj1])^2 #nx8
#        Loss_in += sum(tmp) # 1x1
#        for obj2 in range(0, obj_num)
#            tmp = (mean[c][obj1]-mean[c][obj2])^2 # 1x8
#            Loss_out += 1 / sum(tmp) 
#



import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable
#instance_code.size() = n, c, h, w = batch, 8, h, w
#instance_gt.size()   = n, c, h, w = batch, 1, h, w
#class_gt.size()      = n, c, h, w = batch, 1, h, w
def instance_loss(instance_code, instance_gt, class_gt, class_num, code_bit):
    
    N,C,H,W = instance_code.size()
    Loss = torch.zeros(1)
    Loss_in = torch.zeros(1)
    Loss_out = torch.zeros(1)
    print 'instance code size ', instance_code.size()
    print 'instance gt size ', instance_gt.size()
    print 'class gt size', class_gt.size()
    print instance_gt[0,:,:,:].size()
    for n in range(N):
        #for cls in range(class_num):
        for cls in range(1, class_num):
            Loss_in_cls = torch.zeros(1)
            Loss_out_cls = torch.zeros(1)
            cls_mean = []
            instance_code_ = instance_code[n,:,:,:]#8,h,w
            index_c = class_gt[n,:,:,:].eq(cls)#1,h,w
            index_c_code_shape = index_c.expand_as(instance_code[n,:,:,:]) #8,h,w
            instance_code_c = instance_code[n,:,:,:][index_c_code_shape] #8x,1
            if(len(instance_code_c.size())>0):
                instance_code_c = instance_code_c.view(code_bit, -1) #8,x
                #print instance_code_c

                #find all instance
                for obj in range(2 ** code_bit):
                    index_o = instance_gt[n,:,:,:].eq(obj) #1,h,w
                    index_c_o = torch.mul(index_c, index_o) #1,h,w
                    index_c_o_code_shape = index_c_o.expand_as(instance_code[n,:,:,:]) #8,h,w
                    instance_code_c_o = instance_code[n,:,:,:][index_c_o_code_shape] #8x,1
                    if(len(instance_code_c_o.size()) > 0):
                        instance_code_c_o = instance_code_c_o.view(code_bit, -1) #8,x
                        mean_instance_c_o = instance_code_c_o.mean(dim=1, keepdim=True)
                        var_instance_c_o = instance_code_c_o.var(dim=1, keepdim=True)
                        Loss_in_cls = Loss_in_cls + var_instance_c_o.sum() / code_bit
                        cls_mean.append(mean_instance_c_o)
                    else:
                        break

                obj_num = len(cls_mean)
                if(obj_num > 0):
                    Loss_in_cls = Loss_in_cls / obj_num
                    #print 123, Loss_in_cls
                if(obj_num > 1):
                    for obj1 in range(obj_num):
                        for obj2 in range(obj_num):
                            if obj1 != obj2:
                                Loss_out_cls = Loss_out_cls + (((cls_mean[obj1] - cls_mean[obj2]).abs()).max() - 1) ** 2
                    An2 = obj_num * (obj_num - 1)
                    Loss_out_cls = Loss_out_cls / An2 


            Loss_in = Loss_in + Loss_in_cls;
            Loss_out = Loss_out + Loss_out_cls

    Loss_in = Loss_in / N / class_num;
    Loss_out = Loss_out / N / class_num;
    print "Loss_in ", Loss_in
    print "Loss_out ", Loss_out
    return Loss_in + Loss_out        




def test1():
    instance_code = torch.randn(16,8,100,200)
    instance_code[instance_code>0] = 1
    instance_code[instance_code<1] = 0
    instance_gt = torch.randn(16,1,100,200)
    instance_gt[instance_gt>0] = 1
    instance_gt[instance_gt<1] = 0
    class_gt = torch.randn(16,1,100,200)
    class_gt[class_gt>0] = 1
    class_gt[class_gt<1] = 0

    class_num = 19
    code_bit = 8
    loss = instance_loss(instance_code, instance_gt, class_gt, class_num, code_bit)
    print 'loss is ',loss

def test2():
    instance_code = torch.zeros(16,8,100,200)
    #instance_code[:, 0, 0:50,:] = 1
    instance_code[:, 1, :, 0:100] = 1
    instance_gt = torch.zeros(16,1,100,200)
    instance_gt[:, 0, 0:50, 0:100] = 0
    instance_gt[:, 0, 0:50, 100:200] = 1
    instance_gt[:, 0, 50:100, 0:100] = 2
    instance_gt[:, 0, 50:100,100:200] = 3
    class_gt = torch.zeros(16,1,100,200)

    class_num = 19
    code_bit = 8
    loss = instance_loss(instance_code, instance_gt, class_gt, class_num, code_bit)
    print 'loss is ',loss

#test1()
#test2()


class InstanceLoss(nn.Module):
    def __init__(self, class_num, code_bit):
        super(InstanceLoss, self).__init__()
        self.class_num = class_num
        self.code_bit = code_bit
        return
    
    def forward(self, instance_code, instance_gt, class_gt):
        instance_code_size = instance_code.size()
        instance_gt_size = instance_gt.size()
        class_gt_size = class_gt.size()
        assert instance_code_size[0] == instance_gt_size[0] == class_gt_size[0]
        assert instance_code_size[1] == self.code_bit
        assert instance_gt_size[1] == 1
        assert class_gt_size[1] == 1
        assert instance_code_size[2:] == instance_gt_size[2:] == class_gt_size[2:]

        N, C, H, W = instance_code_size
        Loss = Variable(torch.zeros(1)).cuda()
        Loss_in = Variable(torch.zeros(1)).cuda()
        Loss_out = Variable(torch.zeros(1)).cuda()
        #Loss = torch.zeros(1)
        #Loss_in = torch.zeros(1)
        #Loss_out = torch.zeros(1)
        for n in range(N):
            for cls in range(self.class_num):
                Loss_in_cls = Variable(torch.zeros(1)).cuda()
                Loss_out_cls = Variable(torch.zeros(1)).cuda()
                cls_mean = []
                instance_code_ = instance_code[n,:,:,:]#8,h,w
                index_c = class_gt[n,:,:,:].eq(cls)#1,h,w
                index_c_code_shape = index_c.expand_as(instance_code[n,:,:,:]) #8,h,w
                instance_code_c = instance_code[n,:,:,:][index_c_code_shape] #8x,1
                if(len(instance_code_c.size())>0):
                    instance_code_c = instance_code_c.view(self.code_bit, -1) #8,x
                    #print instance_code_c

                    #find all instance
                    for obj in range(2 ** self.code_bit):
                        index_o = instance_gt[n,:,:,:].eq(obj) #1,h,w
                        index_c_o = torch.mul(index_c, index_o) #1,h,w
                        index_c_o_code_shape = index_c_o.expand_as(instance_code[n,:,:,:]) #8,h,w
                        instance_code_c_o = instance_code[n,:,:,:][index_c_o_code_shape] #8x,1
                        if(len(instance_code_c_o.size()) > 0):
                            instance_code_c_o = instance_code_c_o.view(self.code_bit, -1) #8,x
                            if(len(instance_code_c_o.size()) > 1 and instance_code_c_o.size()[1] > 0):
                                mean_instance_c_o = instance_code_c_o.mean(dim=1, keepdim=True)
                                var_instance_c_o = instance_code_c_o.var(dim=1, keepdim=True, unbiased=False)
                                #print type(Loss_in_cls)
                                #print type(var_instance_c_o.sum() / self.code_bit)
                                Loss_in_cls = Loss_in_cls + var_instance_c_o.sum() / self.code_bit
                                cls_mean.append(mean_instance_c_o)
                        else:
                            break

                    obj_num = len(cls_mean)
                    if(obj_num > 0):
                        Loss_in_cls = Loss_in_cls / obj_num
                        #print 123, Loss_in_cls
                    if(obj_num > 1):
                        for obj1 in range(obj_num):
                            for obj2 in range(obj_num):
                                if obj1 != obj2:
                                    Loss_out_cls = Loss_out_cls + (((cls_mean[obj1] - cls_mean[obj2]).abs()).max() - 1) ** 2
                        An2 = obj_num * (obj_num - 1)
                        Loss_out_cls = Loss_out_cls / An2 


                Loss_in = Loss_in + Loss_in_cls;
                Loss_out = Loss_out + Loss_out_cls

        Loss_in = Loss_in / N / self.class_num;
        Loss_out = Loss_out / N / self.class_num;
        #print "Loss_in ", Loss_in
        #print "Loss_out ", Loss_out
        #return Loss_in + Loss_out        
        return Loss_in #+ 0.01 * Loss_out

