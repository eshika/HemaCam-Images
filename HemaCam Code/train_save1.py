import os, datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *

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

# number of training steps to complete - look at TEnsorboard to check when loss stops decreasing
training_iters = 5000

# interval to display info in terminal
step_display = 100

# interval to save checkpoint 
step_save = 1000

# filepath to save checkpoints - change for each modification to retain old models - can browse on Tensorboard named there according pathname
path_save = './alexnet_bn_no_cluster/'

# All training and validation data for the run comes from here
data_root = '../../Images/cells_new/'

# '' if training new model, else something like './alexnet_bn_no_cluster/-2000' to load existing checkpoint from specified step (2000 in this example)
# './alexnet_bn_no_cluster/-2000'
start_from = '' 


# After convulutions normalizes the batch to ensure each image has normalized baseline so a specific image does not dominate the model for each iteration
def batch_norm_layer(x, train_phase, scope_bn):
    return batch_norm(x, decay=0.9, center=True, scale=True,
    updates_collections=None,
    is_training=train_phase,
    reuse=None,
    trainable=True,
    scope=scope_bn)

# The core model defines the graph and the layers
def alexnet(x, keep_dropout, train_phase):
    weights = {
        'wc1': tf.Variable(tf.random_normal([11, 11, 1, 96], stddev=np.sqrt(2./(11*11)))),
        'wc2': tf.Variable(tf.random_normal([5, 5, 96, 32], stddev=np.sqrt(2./(5*5*96)))),
        'wc3': tf.Variable(tf.random_normal([3, 3, 32, 384], stddev=np.sqrt(2./(3*3*256)))),
        'wc4': tf.Variable(tf.random_normal([3, 3, 384, 32], stddev=np.sqrt(2./(3*3*384)))),
        'wc5': tf.Variable(tf.random_normal([3, 3, 32, 32], stddev=np.sqrt(2./(3*3*32)))),

        'wf6': tf.Variable(tf.random_normal([32, 4096], stddev=np.sqrt(2./(7*7*32)))),
        'wf7': tf.Variable(tf.random_normal([4096, 4096], stddev=np.sqrt(2./4096))),
        'wo': tf.Variable(tf.random_normal([4096, 3], stddev=np.sqrt(2./4096)))
    }

    biases = {
        'bo': tf.Variable(tf.ones(3))
    }

    # Conv + ReLU + Pool, 224->55->27
    image = tf.image.rgb_to_grayscale(x)
    conv1 = tf.nn.conv2d(image, weights['wc1'], strides=[1, 4, 4, 1], padding='SAME')
    conv1 = batch_norm_layer(conv1, train_phase, 'bn1')
    conv1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU  + Pool, 27-> 13
    conv2 = tf.nn.conv2d(pool1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = batch_norm_layer(conv2, train_phase, 'bn2')
    conv2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU, 13-> 13
    conv3 = tf.nn.conv2d(pool2, weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
    conv3 = batch_norm_layer(conv3, train_phase, 'bn3')
    conv3 = tf.nn.relu(conv3)

    # Conv + ReLU, 13-> 13
    conv4 = tf.nn.conv2d(conv3, weights['wc4'], strides=[1, 1, 1, 1], padding='SAME')
    conv4 = batch_norm_layer(conv4, train_phase, 'bn4')
    conv4 = tf.nn.relu(conv4)

    # Conv + ReLU + Pool, 13->6
    conv5 = tf.nn.conv2d(conv4, weights['wc5'], strides=[1, 1, 1, 1], padding='SAME')
    conv5 = batch_norm_layer(conv5, train_phase, 'bn5')
    conv5 = tf.nn.relu(conv5)
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # FC + ReLU + Dropout
    fc6 = tf.reshape(pool5, [-1, weights['wf6'].get_shape().as_list()[0]])
    fc6 = tf.matmul(fc6, weights['wf6'])
    fc6 = batch_norm_layer(fc6, train_phase, 'bn6')
    fc6 = tf.nn.relu(fc6)
    fc6 = tf.nn.dropout(fc6, keep_dropout)
    
    # FC + ReLU + Dropout
    fc7 = tf.matmul(fc6, weights['wf7'])
    fc7 = batch_norm_layer(fc7, train_phase, 'bn7')
    fc7 = tf.nn.relu(fc7)
    fc7 = tf.nn.dropout(fc7, keep_dropout)

    # Output FC
    out = tf.add(tf.matmul(fc7, weights['wo']), biases['bo'])
    
    return out


##### MAIN STARTS HERE

# Construct dataloader
opt_data_train = {
    'data_root': data_root,   
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True
    }
opt_data_val = {
    'data_root': data_root,   
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
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
tf.summary.image('input', x, 5)
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

# EVALUATION STEP - Once model is trained and saved in path_save location
    print('Evaluating the test set')
    file = open(path_save + 'eval_results.txt', 'w')
    testdir = os.path.join(os.getcwd(), 'test')
    for im in os.listdir(testdir):
        if im[0] != '.':
            print(os.path.join(testdir, im))
            path = os.path.join(testdir, im)
            images = loader_val.load_image(path)
            images = np.expand_dims(images, axis=0)
            label = sess.run(tf.nn.top_k(logits, k=3, sorted=True, name=None), feed_dict={x: images, keep_dropout: 1., train_phase: False})
            file.write(path+' '+' '.join([str(num) for num in label.indices[0]])+'\n')
    file.close()
    print('Evaluation Finished! Accuracy Top1 = ' + "{:.4f}".format(acc1_total)





