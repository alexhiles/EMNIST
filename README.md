EMNIST
=====

![](https://raw.githubusercontent.com/Coopss/EMNIST/master/static/preview.gif)

Developed by @coopss

##### Description

This project was intended to explore the properties of convolution neural networks (CNN) and see how they compare to recurrent convolution neural networks (RCNN). This was inspired by a [paper](http://www.cv-foundation.org/openaccess/content_cvpr_2015/app/2B_004.pdf "Recurrent Convolutional Neural Network for Object Recognition") I read that details the effectiveness of RCNNs in object recognition as they perform or even out perform their CNN counterparts with fewer parameters. Aside from exploring CNN/RCNN effectiveness, I built a simple interface to test the more challenging [EMNIST dataset](https://arxiv.org/abs/1702.05373 "EMNIST: an extension of MNIST to handwritten letters") dataset (as opposed to the [MNIST dataset](http://yann.lecun.com/exdb/mnist/ "THE MNIST DATABASE of handwritten digits"))

### Current Implementation
  * Multi-layered CNN used to classify EMNIST dataset.
  * Train on the [byclass dataset](http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip) (*direct download link*). The downloaded .zip file contains multiple MATLAB files which are different datasets according to the EMNIST paper.
  * See the [paper](https://arxiv.org/abs/1702.05373 "EMNIST: an extension of MNIST to handwritten letters") for more information.

## Environment

#### Anaconda: Python 3.5.3
  * Tensorflow or tensorflow-gpu (See [here](https://www.tensorflow.org/install/ "Installing TensorFlow") for more info)
  * Keras
  * Flask
  * Numpy
  * Scipy

  Note: All dependencies for current build can be found in dependencies.txt

## Usage
Once the .zip file containing the datasets is downloaded, place the 'emnist-byclass.mat file in the main directory where "train.py" is located. Then call:

    python train.py --file emnist-byclass.mat

for default train.

#### [train.py](https://github.com/alexhiles/EMNIST/train.py)
A training program for classifying the EMNIST dataset

    usage: train.py [-h] --file [--width WIDTH] [--height HEIGHT] [--max MAX] [--epochs EPOCHS] [--verbose]

##### Required Arguments:

    -f FILE, --file FILE  Path .mat file data

##### Optional Arguments:

    -h, --help            show this help message and exit
    --width WIDTH         Width of the images
    --height HEIGHT       Height of the images
    --max MAX             Max amount of data to use
    --epochs EPOCHS       Number of epochs to train on
    --verbose         Enables verbose printing

#### [evaluate.py](https://github.com/alexhiles/EMNIST/evaluate.py)
A program which loads a trained CNN from file and shows an example case of using it for prediction. The example here is a ones matrix (for demonstrative purposes of predicting).
