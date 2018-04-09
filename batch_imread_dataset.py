#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
the author: leilei
"""

import numpy as np
import os
import cv2
'''
核心：不要一次性全部读取图像到内存，要分批次读取
'''
class Data(object):
    def __init__(self,dataset_path, image_list_path,class_number):
        '''
        txt 内容： 图片名字（无后缀名）eg:
        123
        234
        '''
        self.dataset_path = dataset_path
        self.filenamelist = self.load_txt(image_list_path)
        self.img_num=len(self.filenamelist)
        self.class_number = class_number
        self.batch_offset = 0  
        self.batch_offsetforeval = 0  
        self.epochs_completed = 0
        self.counter = 0
    def load_txt(self,path):
        filenamelist = np.loadtxt(path, dtype=str)
        return filenamelist
    def load_images(self,flag):
        dataset_path = self.dataset_path
        if not os.path.exists(dataset_path):
            print('you need to input valid path to dataset')
        else:
            if flag == 'train':
               img_comp = 'train/sat/'
            if flag == 'valid':
               img_comp = 'valid/sat/'
            ext = '.tif'                 
            self.images = np.array([self.load_origin_image(os.path.join(dataset_path,img_comp)+name+ext) for name in self.filenamelist[self.start:self.end]])
            
        return self.images
    
    def load_labels(self,flag):
        dataset_path = self.dataset_path
        if not os.path.exists(self.dataset_path):
            print('you need to input valid path to dataset')
        else:
            if flag == 'train':
               label_ext = '.tif'
               label_comp = 'train/map/'
            if flag == 'valid':
               label_ext = '.tif'
               label_comp = 'valid/map/'
            self.labels = np.array([self.load_label(os.path.join(dataset_path,label_comp)+name+label_ext) for name in self.filenamelist[self.start:self.end]])
                                                     
        return self.labels
    
    def load_origin_image(self, filename):
        image = cv2.imread(filename)
        return np.array(image)
    
#    def load_label(self, filename):
#        label=cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
#        return np.array(label)

    def load_label(self, filename):
        label=cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
        
        [h,w]=[label.shape[0],label.shape[1]]
        label_=np.zeros(shape=[h,w,self.class_number],dtype=np.int32)
        for i in range(self.class_number):
            label_[:,:,i]=label==i
        return np.array(label_)
    
    def next_batch(self,batch_size,flag='train'):
        self.start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.img_num:
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            perm = np.arange(self.img_num)
            np.random.shuffle(perm)
            if flag=='train' or flag == 'valid':
                self.filenamelist=self.filenamelist[perm]
            self.start = 0
            self.batch_offset = batch_size
        self.end = self.batch_offset
        if flag == 'train' or flag == 'valid':
            imgs=self.load_images(flag)
            labs=self.load_labels(flag)
        return imgs,labs
        
        
