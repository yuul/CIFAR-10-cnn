print("Starting imports")

import keras
print("Keras Imported")
import time
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

# load the model
model = models.model4()
print("Model compiled, training beginning")

# train!!!
start_time = time.time()
model.fit(train_images, train_class, epochs=10, batch_size=256, callbacks=[history], 
    validation_data=(test_images, test_class), verbose=1)
print("--- %s seconds ---" % (time.time() - start_time))  
model.save('model_try3.h5')

print("\nTraining over, evaluation beginning")
score = model.evaluate(test_images, test_class)

print(score)
utils.plot_losses(history.losses)