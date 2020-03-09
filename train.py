import os, pickle, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from keras.layers import Conv2D, MaxPooling2D, Convolution2D, Dropout, Dense, Flatten
from keras.models import Sequential, save_model
from keras.utils import np_utils
from keras import backend as k
from keras.callbacks import TensorBoard

from scipy.io import loadmat

from numpy import save, rot90, fliplr
class Data:
    """ Contains information on any dataset to be trained on
    """
    def __init__(self, filepath, width = 28, height = 28, verbose = False):

        self.filepath = filepath
        self.width    = width
        self.height   = height
        self.verbose  = verbose
        ((self.training_images, self.training_labels), (self.testing_images,   \
        self.testing_labels), self.mapping, self.nb_classes) = self.load_data()
    def load_data(self, max_=None):
        ''' Load data in from .mat file as specified by the paper.

        Arguments:
            mat_file_path: path to the .mat, should be in sample/

        Optional Arguments:
            width: specified width
            height: specified height
            max_: the max number of samples to load
            verbose: enable verbose printing

        Returns:
            A tuple of training and test data, and the mapping for class code to ascii value,
            in the following format:
                - ((training_images, training_labels), (testing_images, testing_labels), mapping)

        '''
        # Local functions
        def rotate(img):
        # Used to rotate images (for some reason they are transposed on read-in)
            flipped = fliplr(img)
            return rot90(flipped)
        """
        """

        # Load convoluted list structure form loadmat
        mat = loadmat(self.filepath)

        # Load char mapping
        mapping = {kv[0]:kv[1:][0] for kv in mat['dataset'][0][0][2]}
        pickle.dump(mapping, open('bin/mapping.p', 'wb' ))

        # Load training data
        if max_ == None:
            max_ = len(mat['dataset'][0][0][0][0][0][0])
        training_images = mat['dataset'][0][0][0][0][0][0][:max_].reshape(max_, self.height, self.width, 1)
        training_labels = mat['dataset'][0][0][0][0][0][1][:max_]

        # Load testing data
        if max_ == None:
            max_ = len(mat['dataset'][0][0][1][0][0][0])
        else:
            max_ = int(max_ / 6)
        testing_images = mat['dataset'][0][0][1][0][0][0][:max_].reshape(max_, self.height, self.width, 1)
        testing_labels = mat['dataset'][0][0][1][0][0][1][:max_]

        # Reshape training data to be valid
        if self.verbose == True: _len = len(training_images)
        for i in range(len(training_images)):
            if self.verbose == True: print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1)/_len) * 100), end='\r')
            training_images[i] = rotate(training_images[i])
        if self.verbose == True: print('')

        # Reshape testing data to be valid
        if self.verbose == True: _len = len(testing_images)
        for i in range(len(testing_images)):
            if self.verbose == True: print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1)/_len) * 100), end='\r')
            testing_images[i] = rotate(testing_images[i])
        if self.verbose == True: print('')

        # Convert type to float32
        training_images = training_images.astype('float32')
        testing_images = testing_images.astype('float32')
        # Normalize to prevent issues with model
        training_images /= 255
        testing_images /= 255

        nb_classes = len(mapping)
        # number of classes

        return ((training_images, training_labels), (testing_images, testing_labels), mapping, nb_classes)
class CNNetwork:
    """ Builds a Sequential model in Keras.
    """
    def __init__(self, Data, nb_filters = 32, pool_size = (2,2), kernel_size = (3,3), \
                 max_ = None, dropout_rate = 0.5):

        self.input_shape  = (Data.height, Data.width, 1)
        # shape of input for first layer
        self.nb_filters   = nb_filters
        # number of convolutional kernels
        self.pool_size    = pool_size
        # dimension of pooling strategy
        self.kernel_size  = kernel_size
        # dimension of convolutional kernels
        self.model       = Sequential()
        # build Keras model
        self.dropout_rate = dropout_rate
        # percentage of highest dropout
        self.verbose      = Data.verbose
        # output flag
        self.nb_classes   = Data.nb_classes
    def __call__(self):
        self.net_architecture()
        # build network architecture
    def net_architecture(self):
        """
        """
        self.model.add(Convolution2D(self.nb_filters,
                            self.kernel_size,
                            padding='valid',
                            input_shape=self.input_shape,
                            activation='relu'))
        # first layer is a convolutional layer
        self.model.add(Convolution2D(self.nb_filters,
                            self.kernel_size,
                            activation='relu'))
        # second layer is a convolutional layer
        self.model.add(MaxPooling2D(pool_size=self.pool_size))
        # third layer is a max pooling layer
        self.model.add(Dropout(self.dropout_rate/4))
        # dropout percentage

        self.model.add(Convolution2D(self.nb_filters,
                            self.kernel_size,
                            padding='valid',
                            activation='relu'))
        # fourth layer is a convolutional layer
        self.model.add(Convolution2D(self.nb_filters,
                            self.kernel_size,
                            activation='relu'))
        # fifth layer is a convolutional layer
        self.model.add(MaxPooling2D(pool_size=self.pool_size))
        # sixth layer is a max pooling layer
        self.model.add(Dropout(self.dropout_rate))
        # dropout neurons
        self.model.add(Flatten())
        # convert to 1D array
        self.model.add(Dense(512, activation='relu'))
        # add fully connected layer
        self.model.add(Dropout(self.dropout_rate))
        # add dropout amount
        self.model.add(Dense(self.nb_classes, activation='softmax'))
        # softmax layer for probabilities
        self.model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
        # use categorical cross entropy loss function for misfit metric, and
        # use adam optimizer to perform gradient descent of the loss
        if self.verbose == True: print(self.model.summary())

        return
class Train:
    """ Takes a Network object in as input, and trains
    """
    def __init__(self, callback = True, batch_size = 256, epochs = 10):
        self.callback   = callback
        #
        self.batch_size = batch_size
        #
        self.epochs     = epochs
        #
    def __call__(self, Data, CNNetwork):
        self.net_train(Data, CNNetwork)
    def net_train(self, Data, CNNetwork):
        """
        """
        # convert class vectors to binary class matrices
        y_train = np_utils.to_categorical(Data.training_labels, Data.nb_classes)
        y_test  = np_utils.to_categorical(Data.testing_labels, Data.nb_classes)

        if self.callback == True:
            # Callback for analysis in TensorBoard
            tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

        M = CNNetwork.model.fit(Data.training_images, y_train,
              batch_size=self.batch_size,
              epochs=self.epochs,
              verbose=1,
              validation_data=(Data.testing_images, y_test),
              callbacks=[tbCallBack] if self.callback else None)



        score = CNNetwork.model.evaluate(Data.testing_images, y_test, verbose=0)
        print('Test score: ', score[0])
        print('Test accuracy: ', score[1])

        self.yaml_save(CNNetwork)
        # Offload model to file
        self.dict_save(M)
        # Offload model to dictionary
        return
    def yaml_save(self, CNNetwork):
        """
        """
        with open("bin/model.yaml", "w") as yaml_file:
            yaml_file.write(CNNetwork.model.to_yaml())
        save_model(CNNetwork.model, 'bin/model.h5')

        return
    def dict_save(self, Model, string = 'model_dictionary'):
        """
        """
        save(string + 'npy', Model)
        #

        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='A training program for classifying the EMNIST dataset')
    parser.add_argument('-f', '--file', type=str, help='Path .mat file data', required=True)
    parser.add_argument('--width', type=int, default=28, help='Width of the images')
    parser.add_argument('--height', type=int, default=28, help='Height of the images')
    parser.add_argument('--max', type=int, default=None, help='Max amount of data to use')
    parser.add_argument('--batch', type=int, default=256, help='How many samples for batched gradient descent')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train on')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enables verbose printing')
    args = parser.parse_args()

    bin_dir = os.path.dirname(os.path.realpath(__file__)) + '/bin'
    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir)

    print('Training begins...')
    D = Data(args.file, width=args.width, height=args.height, verbose=args.verbose)
    # create object containing info on dataset
    N = CNNetwork(D)
    # initialize CNN object with info from data object D
    N()
    # generate convolutional neural network architecture
    T = Train(batch_size = args.batch, epochs = args.epochs)(D, N)
    # train model
