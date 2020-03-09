import os
from keras.models import load_model
from numpy import ones, argmax
bin_dir = os.getcwd() + '/bin'
if os.path.exists(bin_dir + '/model.h5'):
    os.chdir(bin_dir)
    model    = load_model('model.h5')
    example  = ones((1, 28, 28, 1))
    # assuming height = width = 28
    Prediction = model.predict(example)
    print(argmax(Prediction))
else:
    print('Train model by executing train.py.')
