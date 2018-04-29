print("Starting imports")

import keras
print("Keras Imported")
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Flatten, BatchNormalization
#from keras.layers import Conv2D, MaxPooling2D

import models
import data_loader
import utils
print("All packages imported")

# this loads all of the data into train_images, train_class, test_images, test_class
train_images, train_class = data_loader.training_loader()
test_images, test_class = data_loader.load_data("cifar-10-batches-py/test_batch")
print("Data done loading and processing")

# create loss history
history = utils.LossHistory()
print("History list created")
# loads data into minibatches
#(X_train_mini_batch, y_train_mini_batch) = mini_batch(X_train, Y_train, 10000, 0)
#(X_test_mini_batch, y_test_mini_batch) = mini_batch(X_test, Y_test, 5000, 0)

model = models.simpler_model()
print("Model compiled, training beginning")

# train!!!
model.fit(train_images, train_class, epochs=10, batch_size=256, callbacks=[history], 
    validation_data=(test_images, test_class), verbose=1)
print("Training over, evaluation beginning")
score = model.evaluate(test_images, test_class)

# model.save('model_try2.h5')
print(score)

utils.plot_losses(history.losses)