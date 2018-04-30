import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys

n = sys.argv[1]
im1 = cv2.imread('/home/cj/pytorch/pytorch-parsing/ckpt/ins_train/img/0_'+n+'_ins_rgb.png', -1)
im2 = cv2.imread('/home/cj/pytorch/pytorch-parsing/ckpt/ins_train/img/0_'+n+'_gts_rgb.png', -1)

im1[im2[:,:,0]==255,:] = 0
#print im2[:,:,]
im3 = im1 *0.5 + im2 *0.5

plt.figure(1)
plt.imshow(im1)
plt.figure(2)
plt.imshow(im2)
plt.figure(3)
plt.imshow(im3.astype(np.uint8))
plt.show()
