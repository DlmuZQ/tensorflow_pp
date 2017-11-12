# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 15:51:49 2017

@author: ZQ
"""

import tensorflow as tf
import numpy as np

from common_fun import print_info
from read_CNN_data import read_all_train_data,read_test_data

#卷积操作
def conv_op(input_op,name,kh,kw,n_out,dh,dw,p):
    #获取输入特征数
    n_in = input_op.get_shape()[-1].value
    
    with tf.name_scope(name) as scope:
        #卷积核
        kernel = tf.get_variable(scope+"w",
                                 shape = [kh,kw,n_in,n_out],
                                 dtype = tf.float32,
                                 initializer = tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op,kernel,(1,dh,dw,1),padding='SAME')
        bias_init_val = tf.constant(0.0,shape = [n_out],dtype = tf.float32)
        biases = tf.Variable(bias_init_val,trainable = True,name = 'b')
        z = tf.nn.bias_add(conv,biases)
        activation = tf.nn.relu(z,name=scope)
        p += [kernel,biases]
        return activation

#全链接操作   
def fc_op(input_op,name,n_out,p):
    n_in = input_op.get_shape()[-1].value
    
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w",
                                 shape = [n_in,n_out],
                                 dtype = tf.float32,
                                 initializer = tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1,shape = [n_out],
                                         dtype = tf.float32),
                            name = 'b')
        activation = tf.nn.relu_layer(input_op,kernel,biases,name = scope)
        p += [kernel,biases]
        return activation
    
def mpool_op(input_op,name,kh,kw,dh,dw):
    return tf.nn.max_pool(input_op,
                          ksize = [1,kh,kw,1],
                          strides = [1,dh,dw,1],
                          padding = 'SAME',
                          name = name)

#VGGnet网络结构
def inference_op(input_op,keep_prob):
    p = []
    
    conv1_1 = conv_op(input_op,name = "conv1_1",kh = 3,kw = 3,n_out = 64,dh = 1,dw = 1,p = p)
    conv1_2 = conv_op(conv1_1,name = "conv1_2",kh = 3,kw = 3,n_out = 64,dh = 1,dw = 1,p = p)    
    pool1 = mpool_op(conv1_2,name = "pool1",kh = 2,kw = 2,dh = 2, dw = 2)
    
    conv2_1 = conv_op(pool1,name = "conv2_1",kh = 3,kw = 3,n_out = 128,dh = 1,dw = 1,p = p)
    conv2_2 = conv_op(conv2_1,name = "conv2_2",kh = 3,kw = 3,n_out = 128,dh = 1,dw = 1,p = p)    
    pool2 = mpool_op(conv2_2,name = "pool2",kh = 2,kw = 2,dh = 2, dw = 2)
    
    conv3_1 = conv_op(pool2,name = "conv3_1",kh = 3,kw = 3,n_out = 256,dh = 1,dw = 1,p = p)
    conv3_2 = conv_op(conv3_1,name = "conv3_2",kh = 3,kw = 3,n_out = 256,dh = 1,dw = 1,p = p)
    conv3_3 = conv_op(conv3_2,name = "conv3_3",kh = 3,kw = 3,n_out = 256,dh = 1,dw = 1,p = p)    
    pool3 = mpool_op(conv3_3,name = "pool3",kh = 2,kw = 2,dh = 2, dw = 2)
    
    conv4_1 = conv_op(pool3,name = "conv4_1",kh = 3,kw = 3,n_out = 512,dh = 1,dw = 1,p = p)
    conv4_2 = conv_op(conv4_1,name = "conv4_2",kh = 3,kw = 3,n_out = 512,dh = 1,dw = 1,p = p)
    conv4_3 = conv_op(conv4_2,name = "conv4_3",kh = 3,kw = 3,n_out = 512,dh = 1,dw = 1,p = p)    
    pool4 = mpool_op(conv4_3,name = "pool4",kh = 2,kw = 2,dh = 2, dw = 2)
    
    conv5_1 = conv_op(pool4,name = "conv5_1",kh = 3,kw = 3,n_out = 512,dh = 1,dw = 1,p = p)
    conv5_2 = conv_op(conv5_1,name = "conv5_2",kh = 3,kw = 3,n_out = 512,dh = 1,dw = 1,p = p)
    conv5_3 = conv_op(conv5_2,name = "conv5_3",kh = 3,kw = 3,n_out = 512,dh = 1,dw = 1,p = p)    
    pool5 = mpool_op(conv5_3,name = "pool5",kh = 2,kw = 2,dh = 2, dw = 2)
    
    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5,[-1,flattened_shape],name = "resh1")
    
    fc6 = fc_op(resh1,name = "fc6",n_out = 4096,p = p)
    fc6_drop = tf.nn.dropout(fc6,keep_prob,name = "fc6_drop")
    
    fc7 = fc_op(fc6_drop,name = "fc7",n_out = 4096,p = p)
    fc7_drop = tf.nn.dropout(fc7,keep_prob,name = "fc7_drop")
    
    fc8 = fc_op(fc7_drop,name = "fc8",n_out = 1000, p = p)
    softmax = tf.nn.softmax(fc8)
    predictions = tf.argmax(softmax,1)
    return predictions,softmax,fc8,p


def rot_data(x_data,idx):
    x_0 = x_data[idx]
    x_0_90 = x_0
    x_0_180 = x_0
    x_0_270 = x_0
    for i in range(0,224):
        x_0_90[:,:,i] = np.rot90(x_0_90[:,:,i])
        x_0_180[:,:,i] = np.rot90(x_0_180[:,:,i],2)
        x_0_270[:,:,i] = np.rot90(x_0_270[:,:,i],3)
    return x_0,x_0_90,x_0_180,x_0_270

def get_next_batch(x_data,y_data,batch_size = 100):
    x = []
    y = []
    for _ in range(batch_size // 20):
        idx = np.random.randint(0,50)
        x_0,x_9,x_18,x_27 = rot_data(x_data,idx)
            
        x.append(x_0)
        x.append(x_9)
        x.append(x_18)
        x.append(x_27)
        
        y.append(y_data[idx])
        y.append(y_data[idx])
        y.append(y_data[idx])
        y.append(y_data[idx])
        
        idx = idx + 50
        x_0,x_9,x_18,x_27 = rot_data(x_data,idx)
            
        x.append(x_0)
        x.append(x_9)
        x.append(x_18)
        x.append(x_27)
        
        y.append(y_data[idx])
        y.append(y_data[idx])
        y.append(y_data[idx])
        y.append(y_data[idx])
        
        idx = idx + 50 
        x_0,x_9,x_18,x_27 = rot_data(x_data,idx)
            
        x.append(x_0)
        x.append(x_9)
        x.append(x_18)
        x.append(x_27)
        
        y.append(y_data[idx])
        y.append(y_data[idx])
        y.append(y_data[idx])
        y.append(y_data[idx])
        
        idx = idx + 50 
        x_0,x_9,x_18,x_27 = rot_data(x_data,idx)
            
        x.append(x_0)
        x.append(x_9)
        x.append(x_18)
        x.append(x_27)
        
        y.append(y_data[idx])
        y.append(y_data[idx])
        y.append(y_data[idx])
        y.append(y_data[idx])
        
        idx = idx + 50 
        x_0,x_9,x_18,x_27 = rot_data(x_data,idx)
            
        x.append(x_0)
        x.append(x_9)
        x.append(x_18)
        x.append(x_27)
        
        y.append(y_data[idx])
        y.append(y_data[idx])
        y.append(y_data[idx])
        y.append(y_data[idx])
        
        '''
        x.append(x_data[idx + 50])
        y.append(y_data[idx + 50])
        
        x.append(x_data[idx + 100])
        y.append(y_data[idx + 100])
        '''
    x = np.array(x)
    y = np.array(y)
    return x,y
#定义CNN
def create_cnn():
    #x = tf.reshape(X,shape=[-1,15,15,360])
    p = []
    x = tf.reshape(X,shape=[-1,9,9,224])
    #w_c1 = tf.Variable(w_alpha*tf.random_normal([3,3,360,512]))
    '''
    w_c1 = tf.Variable(w_alpha*tf.random_normal([3,3,224,512]))    
    b_c1 = tf.Variable(b_alpha*tf.random_normal([512]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x,w_c1,strides=[1,1,1,1],padding='SAME'),b_c1))
    conv1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv1 = tf.nn.dropout(conv1,keep_prob)
    '''
    conv1_1 = conv_op(x,"conv1_1",kh = 3,kw = 3,n_out = 256,dh = 1,dw = 1,p = p)
    print(conv1_1.get_shape().as_list())
    conv1_2 = conv_op(conv1_1,"conv1_2",kh = 3,kw = 3,n_out = 256,dh = 1,dw = 1,p = p)
    print(conv1_2.get_shape().as_list())
    conv1_3 = conv_op(conv1_2,"conv1_3",kh = 3,kw = 3,n_out = 256,dh = 1,dw = 1,p = p)
    print(conv1_3.get_shape().as_list())
    pool1 = mpool_op(conv1_3,name = "pool1",kh = 2,kw = 2,dh = 2,dw = 2)
    print(pool1.get_shape().as_list())
    '''
    w_c2 = tf.Variable(w_alpha*tf.random_normal([3,3,512,1024]))
    b_c2 = tf.Variable(b_alpha*tf.random_normal([1024]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1,w_c2,strides=[1,1,1,1],padding='SAME'),b_c2))
    conv2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv2 = tf.nn.dropout(conv2,keep_prob)
    '''
    conv2_1 = conv_op(pool1,"conv2_1",kh = 3,kw = 3,n_out = 512,dh = 1,dw = 1,p = p)
    print(conv2_1.get_shape().as_list())
    conv2_2 = conv_op(conv2_1,"conv2_2",kh = 3,kw = 3,n_out = 512,dh = 1,dw = 1,p = p)
    print(conv2_2.get_shape().as_list())
    conv2_3 = conv_op(conv2_2,"conv2_3",kh = 3,kw = 3,n_out = 512,dh = 1,dw = 1,p = p)
    print(conv2_3.get_shape().as_list())
    pool2 = mpool_op(conv2_3,name = "pool2",kh = 2,kw = 2,dh = 2,dw = 2)    
    print(pool2.get_shape().as_list())
    
    '''
    w_c3 = tf.Variable(w_alpha*tf.random_normal([3,3,1024,2048]))
    b_c3 = tf.Variable(b_alpha*tf.random_normal([2048]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2,w_c3,strides=[1,1,1,1],padding='SAME'),b_c3))
    conv3 = tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv3 = tf.nn.dropout(conv3,keep_prob)
    print(conv3.get_shape().as_list())
    '''
    conv3_1 = conv_op(pool1,"conv3_1",kh = 3,kw = 3,n_out = 512,dh = 1,dw = 1,p = p)
    print(conv3_1.get_shape().as_list())
    conv3_2 = conv_op(conv3_1,"conv3_2",kh = 3,kw = 3,n_out = 512,dh = 1,dw = 1,p = p)
    print(conv3_2.get_shape().as_list())
    conv3_3 = conv_op(conv3_2,"conv3_3",kh = 3,kw = 3,n_out = 512,dh = 1,dw = 1,p = p)
    print(conv3_3.get_shape().as_list())
    pool3 = mpool_op(conv3_3,name = "pool3",kh = 2,kw = 2,dh = 2,dw = 2)    
    print(pool3.get_shape().as_list())
    
    #fully connect layer
    '''
    w_d = tf.Variable(w_alpha*tf.random_normal([2*2*2048,2048]))
    b_d = tf.Variable(b_alpha*tf.random_normal([2048]))
    dense = tf.reshape(conv3,[-1,w_d.get_shape().as_list()[0]])
    print(dense.get_shape().as_list())
    dense = tf.nn.relu(tf.add(tf.matmul(dense,w_d),b_d))
    dense = tf.nn.dropout(dense,keep_prob)
    print(dense.get_shape().as_list())
    
    w_out = tf.Variable(w_alpha*tf.random_normal([2048,out_size]))
    b_out = tf.Variable(b_alpha*tf.random_normal([out_size]))
    out = tf.add(tf.matmul(dense,w_out),b_out)
    print(out.get_shape().as_list())
    '''
    shp = pool3.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool3,[-1,flattened_shape],name = "resh1")
    print(resh1.get_shape().as_list())
    
    fc4 = fc_op(resh1,name = "fc4",n_out = 4096,p = p)
    fc4_drop = tf.nn.dropout(fc4,keep_prob,name = "fc4_drop")
    print(fc4_drop.get_shape().as_list())
    
    fc5 = fc_op(fc4_drop,name = "fc5",n_out = 4096,p = p)
    fc5_drop = tf.nn.dropout(fc5,keep_prob,name = "fc5_drop")
    print(fc5_drop.get_shape().as_list())
    
    out = fc_op(fc5_drop,name = "out",n_out = 5,p = p)
    print(out.get_shape().as_list())
    
    return out

def train_cnn(x_all_data,y_all_data):    
    output = create_cnn()
    
    print_info("created cnn ...")
    print_info("start train ...")
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=output))
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.001).minimize(loss)
    
    max_idx_p = tf.argmax(output,1)
    max_idx_l = tf.argmax(Y,1)    
    
    correct_pred = tf.equal(max_idx_p,max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    
    saver = tf.train.Saver(max_to_keep=3)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        step = 0
        while True:
            batch_x,batch_y = get_next_batch(x_all_data,y_all_data)
            _,loss_ = sess.run([optimizer,loss],feed_dict = {X:batch_x,Y:batch_y,keep_prob:0.75})
            print(step,loss_)
            '''
            out = sess.run([output],feed_dict={X:batch_x,Y:batch_y,keep_prob:0.75})
            print("out:",out)
            y = sess.run([Y],{X:batch_x,Y:batch_y,keep_prob:0.75})
            print("y:",y)
            max_p = sess.run([max_idx_p],feed_dict={X:batch_x,Y:batch_y,keep_prob:0.75})
            max_l = sess.run([max_idx_l],feed_dict={X:batch_x,Y:batch_y,keep_prob:0.75})
            print("max_p:",max_p)
            print("max_l",max_l)
            break
            '''
            if step % 10 == 0:
                batch_x_test,batch_y_test = get_next_batch(x_all_data,y_all_data)
                acc = sess.run(accuracy,feed_dict={X:batch_x_test,Y:batch_y_test,keep_prob:1.0})
                print(step,acc)
                if acc > 0.7:
                    saver.save(sess,"./model/cnn.model",global_step=step)
                if acc > 0.8:
                    saver.save(sess,"./model/cnn.model",global_step=step)
                    #break
                if acc > 0.9:
                    saver.save(sess,"./model/cnn.model",global_step=step)
                    break
            step += 1
def test_cnn(x_data):
    
    output = create_cnn()
    l = []
    saver = tf.train.Saver()    
    with tf.Session() as sess:                                       
        saver.restore(sess,"./model/cnn.model-450")        
        preject = tf.argmax(output,1)
        for i in range(len(x_data)//100):
            x_in = x_data[i*100:(i+1)*100]
            x_in = np.array(x_in)
            label = sess.run(preject,feed_dict={X:x_in,keep_prob:1})
            l.append(label)                    
    return l

if __name__ == '__main__':
    train = 1
    if train == 1:
        x_data,y_data = read_all_train_data()       
        
        X = tf.placeholder(tf.float32,[100,9,9,224])
        Y = tf.placeholder(tf.float32,[100,5])
        
        keep_prob = tf.placeholder(tf.float32)
        train_cnn(x_data,y_data)
    if train == 0:
        x_data = read_test_data('E:/Imaging/CNNTest/test_108_aviris_2.txt')        
        tf.reset_default_graph()  
        X = tf.placeholder(tf.float32,[100,9,9,224])        
        keep_prob = tf.placeholder(tf.float32)
        l = test_cnn(x_data)    