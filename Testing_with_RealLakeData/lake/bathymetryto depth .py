# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 15:00:32 2021

@author: KASI VISWANATH
"""

import cv2
import sys
import numpy as np
import pandas as pd

np.set_printoptions(threshold=sys.maxsize)

image=['round.png','tahoe.png','holly_water.png','golden.png','crater.png','irogami.png','jordan.png','pleasant.png','tuttle.png','blue.png','maskenthine.png','wagon_train.png']
hg=[37.1,499,17,7.3,591.9,1.5,25.9,7.3,9.1,6.1,5.5,5.5]


for t in range(12):
    minh=0
    t=3
    maxh=hg[t]

    img=cv2.imread(image[t])
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    h=img.shape[0]
    w=img.shape[1]
    print (h,w)
    # img=cv2.resize(img,(7392,1))
    img=img.flatten()
    pmax=img[np.argmax(img)]
    ## 226 is the highest pixel value
    ## 29 is the lowest pixel value 
    #print(img[f])
    for i in range(h*w):
        if img[i]==0:
            img[i]=255        
    # print(img)
    #print(img.shape)
    pmin=img[np.argmin(img)]
    #print(img[f])
    mask=np.zeros(h*w)
    trans=(maxh)/(pmax)
    for i in range(h*w):
        if img[i]==255:
            continue
        else:
            mask[i]=round(img[i]*trans,1)
    for i in range(h*w):
        if img[i]==255:
            img[i]=0
    #q=np.argmax(mask)
    #print(mask[q])
    mask=np.insert(mask,0,w)
    mask=np.insert(mask,0,h)
    img=np.reshape(img,(h,w))
    mask.tofile(str(t)+'.csv',sep=',')
    print('ok')
    break
    #pd.DataFrame(mask).to_csv(str(t)+'.csv',header=None,index=None)
#mask=np.reshape(mask,(h,w))
#print(mask)
# cv2.imshow('lol',img)
# cv2.waitKey(0)
