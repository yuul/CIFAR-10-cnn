# this file will handle all of the data loading functions
import pickle
import numpy as np

# this sets some constants to use
img_size = 32
num_channels = 3
num_class = 10
num_images = 50000

def unpickle(file):
    """
    Loads the data into a dictionary
    Must be given the correct file path
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

def one_hot_encoder(array, n):
    """
    Takes an array with the shape (length,) and returns an array of shape (length,n)
    N is the number that will be one-hot encoded into
    """
    length = array.shape[0]
    one_hot = np.zeros((length, n+1))
    one_hot[np.arange(length), array] = 1
    one_hot = one_hot[:, 1:n+1]

    return one_hot

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
    # Convert the class numbers to one hot
    one_hot = one_hot_encoder(cls, num_class)

    return images, one_hot

def training_loader():
    """
    This loads all of the data for the training array
    The data is originally split across 5 different files so this will consolidate them into one array
    """

    # Pre-allocate the arrays for the images and class-numbers for efficiency.
    images = np.zeros(shape=[num_images, img_size, img_size, num_channels], dtype=float)
    cls = np.zeros(shape=[num_images, num_class], dtype=int)

    index = 0

    # iterates over all of the files
    for i in range(5):

        # Load the images and class-numbers from the data-file.
        images_batch, cls_batch = load_data(filename="cifar-10-batches-py/data_batch_" + str(i + 1))
        images[index:index+10000, :] = images_batch
        cls[index:index+10000, :] = cls_batch

        index = index + 10000

    return images, cls