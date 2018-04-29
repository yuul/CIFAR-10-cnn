# this provides some utility functions so as to lessen the code in the main script
import keras.callbacks as cb
from matplotlib import pyplot as plt

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