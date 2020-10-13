#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import input_data
mnist = input_data.read_data_sets("./", one_hot=True)

def weight_variable (shape):
    init=tf.random_normal(shape,stddev=0.01)
    return tf.Variable(init)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool22(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x=tf.placeholder(tf.float32, [None,784])
y_=tf.placeholder(tf.float32,[None,10])

x_image=tf.reshape(x,[-1,28,28,1])

w_conv=weight_variable([5,5,1,10])
b_conv=weight_variable([10])

w_fc1=weight_variable([14*14*10,400])
b_fc1=weight_variable([400])

w_fc2=weight_variable([400,10])
b_fc2=weight_variable([10])

h_conv=tf.nn.relu(conv2d(x_image,w_conv)+b_conv)
h_pool=max_pool22(h_conv)
h_pool_flat=tf.reshape(h_pool,[-1,14*14*10])

h_fc1=tf.nn.relu(tf.matmul(h_pool_flat,w_fc1)+b_fc1)
y=tf.nn.softmax(tf.matmul(h_fc1,w_fc2)+b_fc2)

mse=tf.reduce_mean(tf.square(y-y_))
train_step=tf.train.AdamOptimizer(0.01).minimize(mse)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(350):
    batchX,batchY=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batchX, y_:batchY})

    if i%50==0:
        print('Iteration: '+str(i)+'train MSE: ' +str(sess.run(mse,feed_dict={x:batchX,y_:batchY})))

print("Optimization Finished!")

preds=sess.run(y,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
print(preds)
preds=np.argmax(preds,axis=1)
truelabels=np.argmax(mnist.test.labels,axis=1)
errors=0.
ci=0

for i in range (preds.shape[0]):
    if preds[i]!=truelabels[i]:
        if ci<5:
            plt.imshow(mnist.test.images[i].reshape(28,28),cmap='gray')
            plt.show()
            ci=ci+1
            print('Pred: '+str(preds[i])+' TrueLabel: '+ str(truelabels[i]))
        errors=errors+1
print('Test error: '+str(errors/preds.shape[0]))


kernel=sess.run(w_conv)
print(kernel.shape)
f,axes=plt.subplots(2,5)
for i in range(10):
    axes.ravel()[i].imshow(kernel[:,:,0,i],cmap='gray',interpolation='none')
    acts=sess.run(h_conv,feed_dict={x:mnist.test.images[0:100]})

print(acts.shape)
f,axes=plt.subplots(2,5)
index=4
for i in range(10):
    axes.ravel()[i].imshow(acts[index,:,:,i],cmap='gray')

plt.show()

