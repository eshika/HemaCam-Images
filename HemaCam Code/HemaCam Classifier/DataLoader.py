import os
import numpy as np
import scipy.misc

np.random.seed(123)

# Loading data from disk
class DataLoaderDisk(object):
    def __init__(self, **kwargs):

        self.load_size = int(kwargs['load_size'])
        self.fine_size = int(kwargs['fine_size'])
        self.data_mean = np.array(kwargs['data_mean'])
        self.randomize = kwargs['randomize']
        self.data_root = os.path.join(kwargs['data_root'])
        
        if kwargs['training']:
            # read data info from lists
            self.list_im = []
            self.list_lab = []
            
            folders = os.listdir(self.data_root)
            self.folders = folders
            print(folders)
            self.num_labels = len(folders)
            for i in range(len(folders)):
                filepath = os.path.join(self.data_root, folders[i])
                for im in os.listdir(filepath):
                    path = os.path.join(filepath, im)
                    self.list_im.append(path)
                    self.list_lab.append(i)
            self.list_im = np.array(self.list_im, np.object)
            self.list_lab = np.array(self.list_lab, np.int64)
            self.num = self.list_im.shape[0]
            print('# Images found:', self.num)
    
            # permutation
            perm = np.random.permutation(self.num) 
            self.list_im[:, ...] = self.list_im[perm, ...]
            self.list_lab[:] = self.list_lab[perm, ...]
    
            self._idx = 0

    def load_image(self, impath):
        image = scipy.misc.imread(impath)
        image = np.array(image)

        # make image square by padding with white
        h, w, c = image.shape
        if h > w:
            pad = np.ones((h, h-w, c))*255
            image = np.concatenate((image, pad), axis = 1)
        elif w > h:
            pad = np.ones((w-h, w, c))*255
            image = np.concatenate((image, pad), axis = 0)

        # consider centering cells in square 

        # resize square image to specified size
        image = scipy.misc.imresize(image, (self.load_size, self.load_size))
        image = image.astype(np.float32)/255.
        image = image - self.data_mean
        
        # randomly flip image
        if self.randomize:
            flip = np.random.random_integers(0, 1)
            if flip>0:
                image = image[:,::-1,:]
            offset_h = np.random.random_integers(0, self.load_size-self.fine_size)
            offset_w = np.random.random_integers(0, self.load_size-self.fine_size)
        else:
            offset_h = (self.load_size-self.fine_size)//2
            offset_w = (self.load_size-self.fine_size)//2

        return image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size, :]
        
    def next_batch(self, batch_size):
        images_batch = np.zeros((batch_size, self.fine_size, self.fine_size, 3)) 
        labels_batch = np.zeros(batch_size)
        for i in range(batch_size):
            image = self.load_image(self.list_im[self._idx])
           
            images_batch[i, ...] = image
            labels_batch[i, ...] = self.list_lab[self._idx]
            
            self._idx += 1
            if self._idx == self.num:
                self._idx = 0
        
        return images_batch, labels_batch
    
    def size(self):
        return self.num

    def reset(self):
        self._idx = 0
        
    def get_num_labels(self):
        return self.num_labels
