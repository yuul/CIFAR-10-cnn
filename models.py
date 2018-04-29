import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D

# this is VGG model
def VGG_model():
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

def Alex_model():
    
    model = Sequential()
    
    