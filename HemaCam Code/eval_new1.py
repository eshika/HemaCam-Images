# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 16:42:30 2018

@author: eshikasaxena
"""


import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *
from model import alexnet
from train import data_root
from newPipeline import pipeline
import csv

pipeline()

print ('Segmentation complete. Starting screening')

# Dataset Parameters
# Input image size square in pixels
load_size = 32

# determines how much of the image can be clipped to still be relevant for training
# (used in DataLoaderDisk for deciding offset, set to load_size if no offsets desired)
fine_size = 28

c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

learning_rate = 0.001
dropout = 0.5 # Dropout, probability to keep units

# filename to save evaluation output - change for each modification to retain old models - can browse on Tensorboard named there according pathname
path_save = '../HemaCam-Data/Segmented_Cells/Cell_Properties/'

# '' if training new model, else something like './alexnet_bn_no_cluster/-2000' to load existing checkpoint from specified step (2000 in this example)
start_from = './Run_1/-2000'

# folder containing all images to be tested
testdir = '../HemaCam-Data/Segmented_Cells/Cell_Images'


##### MAIN STARTS HERE
if __name__ == "__main__":
    # This is the Graph input
    # x is input images and y is the label for the images
    x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
    y = tf.placeholder(tf.int64, None)
    keep_dropout = tf.placeholder(tf.float32)
    train_phase = tf.placeholder(tf.bool)
    
    # Construct model
    logits = alexnet(x, keep_dropout, train_phase)
    
    # Define loss and optimizer
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
    train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    
    # Evaluate model
    accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 1), tf.float32))
    
    # define initialization
    init = tf.global_variables_initializer()
    
    # define saver for checkpoint
    saver = tf.train.Saver()
    
    opt_data_test = {
        'data_root': testdir,   
        'load_size': load_size,
        'fine_size': fine_size,
        'data_mean': data_mean,
        'randomize': False,
        'training': False
        }
    
    loader_test = DataLoaderDisk(**opt_data_test)
    
    # Launch the graph in the session
    with tf.Session() as sess:
        # Initialization
        if len(start_from)>1:
            # THIS IS WHERE THE TRAINED MODEL IS LOADED
            saver.restore(sess, start_from)
            step = int(start_from.split('-')[-1])
            
        # EVALUATION STEP - Once model is trained and saved in path_save location
            print('Evaluating the test set')
            label_map = os.listdir(data_root)
            csv_files = os.listdir(path_save)
            lines = []

            for path in csv_files:
                if '.csv' in path:
                    csv_file = os.path.join(path_save, path)
                    with open(csv_file, 'r+') as file:
                        reader = csv.reader(file)
                        for row in reader:
                            if len(row) != 0 and '.png' in row[0]:
                                path = os.path.join(testdir, row[0])
                                images = loader_test.load_image(path)
                                images = np.expand_dims(images, axis=0)
                                confidence_values = sess.run(tf.nn.softmax(logits), feed_dict={x: images, keep_dropout: 1., train_phase: False}).flatten()

                                maxConfidence = round(max(confidence_values), 2)
                                label = label_map[np.argmax(confidence_values)]
                                lines.append(row+[label, maxConfidence])
                            elif len(row) != 0:
                                header = row+['Label', 'Confidence']

            output_file = open(path_save+'results.csv', 'w')
            writer = csv.writer(output_file)
            writer.writerow(header)
            writer.writerows(lines)
            output_file.close()

            print('Evaluation Finished!')
            
        else:
            print('No model found')
    
    
    
    
    
