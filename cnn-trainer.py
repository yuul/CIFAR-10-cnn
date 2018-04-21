import pickle
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras.callbacks as cb
from matplotlib import pyplot as plt

# Width and height of each image.
img_size = 32

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Number of classes
num_class = 10

def unpickle(file):
    """
    Loads the data into a dictionary
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def convert_images(raw):
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """

    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images

def load_data(filename):
    """
    Load a pickled data-file from the CIFAR-10 data-set
    and return the converted images and the class-number for each image.
    """

    # Load the pickled data-file.
    data = unpickle(filename)
    # Get the raw images.
    raw_images = data[b'data']
    # Get the class-numbers for each image. Convert to numpy-array.
    cls = np.array(data[b'labels'])
    # Convert the images.
    images = convert_images(raw_images)

    return images, cls

def one_hot_encoder(array, n):
    """
    Takes an array with the shape (length,) and returns an array of shape (length,n)
    N is the number that will be one-hot encoded into
    """
    length = array.shape[0]
    one_hot = np.zeros((length, n+1))
    one_hot[np.arange(length), array] = 1
    
    return one_hot

images1, cls1 = load_data('cifar-10-batches-py/data_batch_1')
print(images1.shape)
print(cls1.shape)

Y_vec = one_hot_encoder(cls1, num_class)