import keras
from keras.datasets import cifar10

def load_data():
    # load images from keras.datasets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # scale the training and test data to grey scale
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    
    # Need to encode are target
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    return (X_train, y_train), (X_test, y_test)