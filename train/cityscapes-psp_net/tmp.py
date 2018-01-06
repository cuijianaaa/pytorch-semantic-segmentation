import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure, color
#im = cv2.imread('/media/cj/Elements/cityscapes_ins/gtFine_trainvaltest/gtFine/train/aachen/aachen_000154_000019_gtFine_instanceIds.png', -1)
im = cv2.imread('/media/cj/Elements/cityscapes_ins/gtFine_trainvaltest/gtFine/train/aachen/aachen_000154_000019_gtFine_labelIds.png', -1)
#print im.dtype
#im = im / 1000
#im = im.astype(np.uint8) 
for i in range(256):
    n = np.sum(im == i)
    if n > 0:
        print '%d number: %d'%(i,n)
dst = color.label2rgb(im)

print dst.shape
plt.imshow(dst)
plt.show()