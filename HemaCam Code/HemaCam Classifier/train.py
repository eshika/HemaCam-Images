import os, datetime
import numpy as np
import tensorflow as tf
from DataLoader import *
from model import alexnet

# Dataset Parameters
# how many images to train on in one step
batch_size = 32

# Input image size square in pixels
load_size = 32

# determines how much of the image can be clipped to still be relevant for training
# (used in DataLoaderDisk for deciding offset, set to load_size if no offsets desired)
fine_size = 28

# number of channels in images (r,g,b)
c = 3

# used to normalize images in DataLoader
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate = 0.001
dropout = 0.5 # Dropout, probability to keep units

# number of training steps to complete - look at TEnsorboard to check when loss stops decreasing
training_iters = 5000

# interval to display info in terminal
step_display = 100

# interval to save checkpoint 
step_save = 1000

# filepath to save checkpoints - change for each modification to retain old models - can browse on Tensorboard named there according pathname
path_save = './Run_esh1/'

# All training and validation data for the run comes from here
data_root = '../../Images/cells_new/'

# '' if training new model, else something like './alexnet_bn_no_cluster/-2000' to load existing checkpoint from specified step (2000 in this example)
# './alexnet_bn_no_cluster/-2000'
start_from = '' 

##### MAIN STARTS HERE
if __name__ == "__main__":
    # Construct dataloader
    opt_data_train = {
        'data_root': data_root,   
        'load_size': load_size,
        'fine_size': fine_size,
        'data_mean': data_mean,
        'randomize': True,
        'training': True
        }
    opt_data_val = {
        'data_root': data_root,   
        'load_size': load_size,
        'fine_size': fine_size,
        'data_mean': data_mean,
        'randomize': False,
        'training': True
        }
    
    loader_train = DataLoaderDisk(**opt_data_train)
    loader_val = DataLoaderDisk(**opt_data_val)
    
    
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
    
    
    # write certain variables to Tensorboard summary for easy model debugging
    # Accuracy should go up to 1 if train and test diverge then its overfitting
    # Loss should be as close to 0 as possible and should be decreasing in the train if loss stops decreasing then model is not learning
    # Print image to Tensorboard
    tf.summary.scalar('accuracy1', accuracy1)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.image('input', x, 20) # Specify the number of images to display in Tensorboard
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(path_save + '/train', tf.get_default_graph())
    test_writer = tf.summary.FileWriter(path_save + '/test', tf.get_default_graph())
    
    # define initialization
    init = tf.global_variables_initializer()
    
    # define saver for checkpoint
    saver = tf.train.Saver()
    
    
    # Launch the graph in the session
    with tf.Session() as sess:
        # Initialization
        if len(start_from)>1:
            # THIS IS WHERE THE TRAINED MODEL IS LOADED
            saver.restore(sess, start_from)
            step = int(start_from.split('-')[-1])
        else:
            sess.run(init)
            step = 0
    
        while step < training_iters:
            # Load a batch of training data
            images_batch, labels_batch = loader_train.next_batch(batch_size)
            
            if step % step_display == 0:
                print('[%s]:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    
                # Calculate batch loss and accuracy on training set
                l, acc1, summary = sess.run([loss, accuracy1, merged], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1.0, train_phase: False}) 
                print("-Iter " + str(step) + ", Training Loss= " + \
                      "{:.4f}".format(l) + ", Accuracy Top1 = " + \
                      "{:.2f}".format(acc1))
                train_writer.add_summary(summary, step)
    
                # Calculate batch loss and accuracy on validation set
                images_batch_val, labels_batch_val = loader_val.next_batch(batch_size)    
                l, acc1, summary = sess.run([loss, accuracy1, merged], feed_dict={x: images_batch_val, y: labels_batch_val, keep_dropout: 1., train_phase: False}) 
                print("-Iter " + str(step) + ", Validation Loss= " + \
                      "{:.4f}".format(l) + ", Accuracy Top1 = " + \
                      "{:.2f}".format(acc1))
                test_writer.add_summary(summary, step)
            
            # Run optimization op (backprop)
            sess.run(train_optimizer, feed_dict={x: images_batch, y: labels_batch, keep_dropout: dropout, train_phase: True})
            
            step += 1
            
            # Save model
            if step % step_save == 0:
                saver.save(sess, path_save, global_step=step)
                print("Model saved at Iter %d !" %(step))
            
        print("Optimization Finished!")
        # TRAINING IS NOW COMPLETE model is in path_save
    
    
    
    
    
    
