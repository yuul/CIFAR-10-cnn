print("Starting imports")
import pickle
import numpy as np
print("Numpy Imported")
import tensorflow as tf
print("Tensorflow Imported")
import keras
print("Keras Imported")
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import keras.callbacks as cb
from matplotlib import pyplot as plt
print("All packages imoported")

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

# this is the model I will be using
def init_model():
    
    ### This code is for my VGG-16 architecture
    model = Sequential()
    
    # block 1
    model.add(Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_normal', input_shape=(img_size, img_size, num_channels)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    
    # block 2
    model.add(Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    
    # block 3
    model.add(Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

    # block 4
    model.add(Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

    # block 5
    model.add(Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
        
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(num_class, activation='softmax', kernel_initializer='he_normal'))
    
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    return model

# not quite sure how this works but understand that it records the history
class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)

# plots stuff
def plot_losses(losses):
        plt.plot(losses)
        plt.title('Loss per batch')
        plt.show()

# this loads all of the data into train_images, train_class, test_images, test_class
train_images, train_class = training_loader()
test_images, test_class = load_data("cifar-10-batches-py/test_batch")
print("Data done loading and processing")

# create loss history
history = LossHistory()
print("History list created")
# loads data into minibatches
#(X_train_mini_batch, y_train_mini_batch) = mini_batch(X_train, Y_train, 10000, 0)
#(X_test_mini_batch, y_test_mini_batch) = mini_batch(X_test, Y_test, 5000, 0)

model = init_model()
print("Model compiled, training beginning")

# train!!!
model.fit(train_images, train_class, epochs=20, batch_size=64, callbacks=[history], 
    validation_data=(test_images, test_class), verbose=1)
print("Training over, evaluation beginning")
score = model.evaluate(test_images, test_class, batch_size=16)

model.save('model_try1.h5')
print(score)
plot_losses(history.losses)