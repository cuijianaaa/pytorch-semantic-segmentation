import cv2
import matplotlib.pyplot as plt
import numpy as np
n = 2
im1 = cv2.imread('/home/cj/pytorch/pytorch-parsing/ckpt/ins_train/img/0_%d_ins_rgb.png'%n, -1)
im2 = cv2.imread('/home/cj/pytorch/pytorch-parsing/ckpt/ins_train/img/0_%d_gts_rgb.png'%n, -1)

im3 = im1 *0.5 + im2 *0.5
plt.imshow(im3.astype(np.uint8))
plt.show()