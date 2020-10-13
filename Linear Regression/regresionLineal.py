#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

#Eliminar mensajes molestos de CUDA
import os
from tensorflow.python.framework import ops
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
ops.reset_default_graph()

# Crea 1000 puntos segun la funcion y=0.1 * x + 0.4 (i.e. y = W * x + b)
num_points = 1000
vectors_set = []
ntrains=10;
learning_factor=0.6;

for i in range(num_points):
    W = 0.1  # W
    b = 0.4  # b
    x1 = np.random.normal(0.0, 1.0)
    nd = np.random.normal(0.0, 0.05)
    y1 = W * x1 + b
    # Se agrega ruido gaussiano
    y1 = y1 + nd
    # Se crean los vectores de aprendizaje
    vectors_set.append([x1, y1])


# Se separa el vector x y el y
x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

# Prediccion o inferencia
with tf.name_scope("LinearRegression") as scope:
	W = tf.Variable(tf.zeros([1]))
	b = tf.Variable(tf.zeros([1]))
	y = W * x_data + b

# Loss function
with tf.name_scope("LossFunction") as scope:
    loss = tf.reduce_mean(tf.square(y - y_data))
# Optimizer
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_factor)
# Training
train = optimizer.minimize(loss)

init = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.Session()
sess.run(init)

# Escribe el archivo de Summary
writer_tensorboard = tf.compat.v1.summary.FileWriter('./LReg_logs/', tf.compat.v1.get_default_graph())

for i in range(ntrains):
	sess.run(train)
	print(i, sess.run(W), sess.run(b), sess.run(loss))

