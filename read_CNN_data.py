# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 10:46:15 2017

@author: ZQ
"""

import numpy as np
import sklearn.preprocessing as prep

from common_fun import print_info

def read_signal_txt(file_name,row,col,channel):
    img = np.zeros((row,col,channel))
    band = 0
    row = 0
    #l = 0
    with open(file_name,'r') as file_open:
        #需要跳过的前几行        
        step = 5
        for line in file_open:
            if step != 0:
                step = step - 1
                continue
            if line == '\n':
                continue
            
            line = line.strip().split('  ')
            #print(line)
            c = 0
            for i in range(len(line)):                
                if line[i] == '':
                    continue
                img[row,c,band] = float(line[i])
                c = c + 1
            
            band = band + 1
            if band == channel:
                band = 0
                row = row + 1
            #print(l)
            #l = l + 1
    return img
            
def read_all_train_data():
    x_data = []
    y_data = []
    data_dir = ('E:/Imaging/CNNROI/9_%d_%d.txt')
    for i in range(5):
        for j in range(1,51):
            cur_txt = data_dir%(i,j)
            cur_data = read_signal_txt(cur_txt,9,9,224)
            if i == 0:
                t = [1,0,0,0,0]
            if i == 1:
                t = [0,1,0,0,0]
            if i == 2:
                t = [0,0,1,0,0]
            if i == 3:
                t = [0,0,0,1,0]
            if i == 4:
                t = [0,0,0,0,1]
            y_data.append(t)
            x_data.append(cur_data)
    print_info("read done!")
    x_data = np.array(x_data,dtype = np.float32)
    y_data = np.array(y_data,dtype = np.float32)
    return x_data,y_data

#样本旋转，产生更多的样本
#策略：central_around指以中心点为中心旋转四周，central是指以中心3*3旋转，每种方式产生3组数据
def rotate_data(x_data,y_data,central_around = 0,central = 0):
    pass

#读取测试数据,返回的数据是list类型
def read_test_data(file_in):
    print_info("start read test data...")
    data = read_signal_txt(file_in,100,100)
    print_info("end read test data...")
    test_data = []
    x,y,z = data.shape
    for j in range(15,y+1):
        for i in range(15,x+1):
            test_cur = data[i-15:i,j-15:j,:]
            test_data.append(test_cur)
    #test_data = np.array(test_data)    
    return test_data

def stander_scale(x_train,x_test):
    preprocessor = prep.StandardScaler().fit(x_train)
    x_train = preprocessor.transform(x_train)
    if x_test is not None:
        x_test = preprocessor.transform(x_test)
    return x_train,x_test
            
if __name__ == '__main__':
    #read_signal_txt('E:/Imaging/CNNROI/15_0_1.txt')
    x_data,y_data = read_all_train_data()
    #test_data = read_test_data('E:/Imaging/ROI_test/test_roi_2_100.txt')
            