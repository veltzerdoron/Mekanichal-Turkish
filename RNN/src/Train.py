# -*- coding: utf-8 -*-
from __future__ import print_function
'''
Created on Jul 17, 2016

@author: Veltzer Doron
'''

import os

import numpy

import keras.models
import keras.layers
import keras.optimizers

import MineRNN

import TurkishPhonology

#MODEL_NAME = 'NN128X8'
#MODEL_NAME = 'LSTM32X3'
#MODEL_NAME = 'LSTM128X2'
#MODEL_NAME = 'RNN128X2'
#MODEL_NAME = 'RNN128X2'
#MODEL_NAME = 'RNN512X2'
MODEL_NAME = 'MINE512x2'

FEATURES = 'FEATURES'
FULL = ''

MODEL_NAME = MODEL_NAME + FEATURES + FULL

BATCH_SIZE = 16
NB_EPOCHS = 25
#NUM_MODELS = 20
NUM_MODELS = 100

def main():
    # Load the train data set
    dataset = numpy.loadtxt('csv/train' + FEATURES + FULL + '.csv', delimiter = ',', dtype = numpy.int)
    
    if not os.path.exists(MODEL_NAME):
        os.makedirs(MODEL_NAME)
    if not os.path.exists(MODEL_NAME + '/weights'):
        os.makedirs(MODEL_NAME + '/weights')
    if not os.path.exists('csv/' + MODEL_NAME + 'testResults'):
        os.makedirs('csv/' + MODEL_NAME + 'testResults')
    if not os.path.exists('csv/' + MODEL_NAME + 'nonceResults'):
        os.makedirs('csv/' + MODEL_NAME + 'nonceResults')
    
    # features or bit network
    #maxPhonemes = dataset.shape[1] - 1
    maxPhonemes = dataset.shape[1] - TurkishPhonology.featuresNum
    
    X = numpy.array(dataset[:, 0:maxPhonemes])
    #Y = numpy.array(dataset[:, -1] - 1)
    y1 = (numpy.array(dataset[:, maxPhonemes + 0] - 1).astype(float) / (numpy.max(dataset[:, maxPhonemes + 0]) - 1)).reshape(-1, 1)
    y2 = (numpy.array(dataset[:, maxPhonemes + 1] * 0).astype(float) / (numpy.max(dataset[:, maxPhonemes + 1]) - 1)).reshape(-1, 1)
    y3 = (numpy.array(dataset[:, maxPhonemes + 2] - 1).astype(float) / (numpy.max(dataset[:, maxPhonemes + 2]) - 1)).reshape(-1, 1)
    Y = numpy.concatenate([y1, y2, y3], 1)
    
    #model = load_model(MODEL_NAME + '/model.h5')
    
    # Add Layers
    HIDDENS = 512
    
    # Define and Compile model
    model = keras.models.Sequential()
    
    model.add(keras.layers.Embedding(input_dim = TurkishPhonology.maxIndex + 1,
                                     input_length = maxPhonemes,
                                     output_dim = HIDDENS,
                                     dropout = 0.1))
    model.add(keras.layers.SimpleRNN(input_dim = HIDDENS,
                                     output_dim = HIDDENS / 8,
                                     activation = 'sigmoid',
                                     dropout_U = 0.1,
                                     dropout_W = 0.1,
                                     return_sequences = True))
    #model.add(keras.layers.LSTM(input_dim = HIDDENS,
    #                            output_dim = HIDDENS / 4,
    #                            dropout_U = 0.1,
    #                            dropout_W = 0.1,
    #                            return_sequences = True))
    model.add(keras.layers.SimpleRNN(input_dim = HIDDENS / 8,
                                     output_dim = HIDDENS / 64,
                                     activation = 'sigmoid',
                                     dropout_U = 0.1,
                                     dropout_W = 0.1,
                                     return_sequences = True))
    #model.add(keras.layers.LSTM(input_dim = HIDDENS / 4,
    #                            output_dim = HIDDENS /16,
    #                            dropout_U = 0.1,
    #                            dropout_W = 0.1,
    #                            return_sequences = True))
    model.add(keras.layers.SimpleRNN(input_dim = HIDDENS / 64,
                                     output_dim = 3,
                                     activation = 'sigmoid',
                                     dropout_U = 0.1,
                                     dropout_W = 0.1,
                                     return_sequences = True))
    #model.add(keras.layers.LSTM(input_dim = HIDDENS / 16,
    #                            output_dim = HIDDENS / 64,
    #                            dropout_U = 0.1,
    #                            dropout_W = 0.1,
    #                            return_sequences = False))
    model.add(MineRNN.MineRNN(input_dim = 3,
                              output_dim = 3,
                              activation = 'sigmoid',
                              dropout_U = 0.1,
                              dropout_W = 0.1,
                              return_sequences = False))
    #model.add(keras.layers.LSTM(input_dim = HIDDENS / 64,
    #                            output_dim = HIDDENS / 256,
    #                            dropout_U = 0.1,
    #                            dropout_W = 0.1,
    #                            return_sequences = False))
    #model.add(keras.layers.SimpleRNN(input_dim = HIDDENS / 256,
    #                                 output_dim = 3,
    #                                 activation = 'sigmoid',
    #                                 dropout_U = 0.1,
    #                                 dropout_W = 0.1,
    #                                 return_sequences = False))
    #model.add(keras.layers.LSTM(input_dim = HIDDENS / 256,
    #                            output_dim = 3,
    #                            dropout_U = 0.1,
    #                            dropout_W = 0.1,
    #                            return_sequences = False))
    #model.add(keras.layers.Dense(1024, input_shape = [maxPhonemes], activation = 'sigmoid'))
    #model.add(keras.layers.Dropout(0.1))
    #model.add(keras.layers.Dense(256, activation = 'sigmoid'))
    #model.add(keras.layers.Dropout(0.1))
    #model.add(keras.layers.Dense(64, activation = 'sigmoid'))
    #model.add(keras.layers.Dropout(0.1))
    #model.add(keras.layers.Dense(16, activation = 'sigmoid'))
    #model.add(keras.layers.Dropout(0.1))
    #model.add(keras.layers.Dense(4, activation = 'sigmoid'))
    #model.add(keras.layers.Dropout(0.1))
    #model.add(keras.layers.Dense(1, activation = 'sigmoid'))
    
    # compile
    rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    model.compile(loss = keras.metrics.mean_squared_error, optimizer = rmsprop, metrics = ['accuracy']) 
    
    # save the model
    model.save(MODEL_NAME + '/model.h5')
    
    # Set apart 10% for validation data that we never train over
    split_at = len(X) * 9 / 10
    #comment the following two lines and uncomment the shuffle in the training loop to have the input shuffled
    
    # Fit the models to the lexicon
    modelNum = 0
    startupWeights = model.get_weights()
    startupWeightsNotReset = True
    requiredAcc = 0.96
    attempt = 0
    while modelNum < NUM_MODELS:
        # generate the modelNum model with a new random seed
        #seed = modelNum
        #numpy.random.seed(seed)
        numpy.random.seed()
        maxAcc = 0
        attempt += 1
        #model.load_weights(MODEL_NAME + '/weights/' + str(modelNum) + '.h5')
        for iteration in xrange(1000):
            print()
            print('-' * 50)
            print('Iteration', iteration)
            
            # Shuffle (X, Y) in unison for train and validation
            indices = numpy.arange(len(X))
            numpy.random.shuffle(indices)
            X_train = X[indices[:split_at]]
            Y_train = Y[indices[:split_at]]
            X_val = X[indices[split_at:]]
            Y_val = Y[indices[split_at:]]
            
            # Train the model
            model.fit(X_train, Y_train,
                      nb_epoch = NB_EPOCHS,
                      batch_size = BATCH_SIZE,
                      #validation_split = 0.1
                      validation_data = (X_val, Y_val))
            
            # Evaluate the model
            acc = model.evaluate(X, Y, batch_size = BATCH_SIZE)[1]
            print('Test acc:', acc)
            if acc > maxAcc:
                maxAcc = acc
            if acc > requiredAcc:
                # save this speaker model and continue
                model.save_weights(MODEL_NAME + '/weights/' + str(modelNum) + '.h5')
                Y_ = model.predict(X, batch_size = BATCH_SIZE)
                numpy.savetxt('csv/' + MODEL_NAME + 'testResults/' + str(modelNum) + '.csv', delimiter = ',', X = Y_, fmt = '%f')
                modelNum = modelNum + 1
                break
            else:
                if acc > 0.8 and startupWeightsNotReset:
                    # If we are half way through save the weights
                    startupWeights = model.get_weights()
                    startupWeightsNotReset = False
        # reset the model weights to initialization state
        model.set_weights(startupWeights)

if __name__ == '__main__':
    main()