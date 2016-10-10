# -*- coding: utf-8 -*-
from __future__ import print_function
'''
Created on Jul 17, 2016

@author: Veltzer Doron
'''

import numpy

import keras.models
import keras.layers
import keras.optimizers

import MineRNN
import Train
import TurkishPhonology

from keras.models import load_model

def main():
    
    if (Train.FULL == ''):
        maxPhonemes = 5
    else:
        maxPhonemes = 15
    
    # Load the dataset to be analyzed
    #dataset = numpy.loadtxt('csv/train' + Train.FEATURES + Train.FULL + '.csv', delimiter = ",", dtype = numpy.int)
    dataset = numpy.loadtxt('csv/test' + Train.FEATURES + Train.FULL + '.csv', delimiter = ",", dtype = numpy.int)
    
    #X = numpy.array(dataset[:, 0:maxPhonemes * TurkishPhonology.featuresNum]).reshape(-1, maxPhonemes, TurkishPhonology.featuresNum)
    X = numpy.array(dataset[:, 0:maxPhonemes]).reshape(-1, maxPhonemes)
    
    # record the relevant statistical behavior of the speaker models
    
    #model = load_model(Train.MODEL_NAME + '/model.h5')
    
    HIDDENS = 512

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
    model.add(keras.layers.SimpleRNN(input_dim = HIDDENS / 8,
                                     output_dim = HIDDENS / 64,
                                     activation = 'sigmoid',
                                     dropout_U = 0.1,
                                     dropout_W = 0.1,
                                     return_sequences = True))
    model.add(keras.layers.SimpleRNN(input_dim = HIDDENS / 64,
                                     output_dim = 3,
                                     activation = 'sigmoid',
                                     dropout_U = 0.1,
                                     dropout_W = 0.1,
                                     return_sequences = True))
    model.add(MineRNN.MineRNN(input_dim = 3,
                              output_dim = 3,
                              activation = 'sigmoid',
                              dropout_U = 0.1,
                              dropout_W = 0.1,
                              return_sequences = False))
    
    # compile
    rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    model.compile(loss = keras.metrics.mean_squared_error, optimizer = rmsprop, metrics = ['accuracy']) 
    
    # analysisRaw includes Alternating, Length, POAIndex(1-4), Height, BackVowelIndex (both of preceding vowel)
    analysisRaw = numpy.zeros(shape = [len(X), 4 + Train.NUM_MODELS])
    for i in xrange(len(X)):
        x = X[i]
        analysisRaw[i, 0] = TurkishPhonology.lengthIndices(x)
        alternatingConsonant = x[maxPhonemes - 1]
        analysisRaw[i, 1] = TurkishPhonology.POAIndex(alternatingConsonant)
        if analysisRaw[i, 0] != 4:
            previouseVowelIndex = x[maxPhonemes - 2]
        else:
            previouseVowelIndex = x[maxPhonemes - 3]
        analysisRaw[i, 2] = TurkishPhonology.HighVowelIndex(previouseVowelIndex)
        analysisRaw[i, 3] = TurkishPhonology.BackVowelIndex(previouseVowelIndex)

    for modelNum in xrange(Train.NUM_MODELS):
        model.load_weights(Train.MODEL_NAME + '/weights/' + str(modelNum) + '.h5')
        Y = model.predict(X, batch_size = Train.BATCH_SIZE)
        #numpy.savetxt('csv/' + Train.MODEL_NAME + 'testResults/' + str(modelNum) + '.csv', delimiter = ',', X = Y, fmt = "%f")
        numpy.savetxt('csv/' + Train.MODEL_NAME + 'nonceResults/' + str(modelNum) + '.csv', delimiter = ',', X = Y, fmt = "%f")
        for i in xrange(len(Y)):
            y = Y[i]
            #analysisRaw[i, 4 + modelNum] = max(0, min(1, round(y)))
            analysisRaw[i, 4 + modelNum] = max(0, min(1, round(y[2])))
    
    # count the alternations per length POA, for total high and back vowels
    analysis = numpy.zeros(shape = [maxPhonemes - 2, 4, 8])
    for i in xrange(len(X)):
        for modelNum in xrange(Train.NUM_MODELS):
            # count total produced words 
            analysis[analysisRaw[i, 0] - 3, analysisRaw[i, 1], 0] = analysis[analysisRaw[i, 0] - 3, analysisRaw[i, 1], 0] + 1
            if analysisRaw[i, 2]:
                # count total high words 
                analysis[analysisRaw[i, 0] - 3, analysisRaw[i, 1], 1] = analysis[analysisRaw[i, 0] - 3, analysisRaw[i, 1], 1] + 1
            if analysisRaw[i, 3]:
                # count total back words 
                analysis[analysisRaw[i, 0] - 3, analysisRaw[i, 1], 2] = analysis[analysisRaw[i, 0] - 3, analysisRaw[i, 1], 2] + 1
            if (analysisRaw[i, 2] and analysisRaw[i, 3]):
                # count total high back words 
                analysis[analysisRaw[i, 0] - 3, analysisRaw[i, 1], 3] = analysis[analysisRaw[i, 0] - 3, analysisRaw[i, 1], 3] + 1

            # handle alternations
            if analysisRaw[i, 4 + modelNum]:
                # count alternating words 
                analysis[analysisRaw[i, 0] - 3, analysisRaw[i, 1], 4] = analysis[analysisRaw[i, 0] - 3, analysisRaw[i, 1], 4] + 1
                if analysisRaw[i, 2]:
                    # count alternating high words 
                    analysis[analysisRaw[i, 0] - 3, analysisRaw[i, 1], 5] = analysis[analysisRaw[i, 0] - 3, analysisRaw[i, 1], 5] + 1
                if analysisRaw[i, 3]:
                    # count alternating back words 
                    analysis[analysisRaw[i, 0] - 3, analysisRaw[i, 1], 6] = analysis[analysisRaw[i, 0] - 3, analysisRaw[i, 1], 6] + 1
                if (analysisRaw[i, 2] and analysisRaw[i, 3]):
                    # count alternating high back words 
                    analysis[analysisRaw[i, 0] - 3, analysisRaw[i, 1], 7] = analysis[analysisRaw[i, 0] - 3, analysisRaw[i, 1], 7] + 1

    print('total alternations')
    print('total')
    print(100 * numpy.sum(analysis[:, :, 4]) / numpy.sum(analysis[:, :, 0]))
    #print('high Vowels')
    #print(100 * numpy.sum(analysis[:, :, 5]) / numpy.sum(analysis[:, :, 1]))
    #print('back Vowels')
    #print(100 * numpy.sum(analysis[:, :, 6]) / numpy.sum(analysis[:, :, 2]))
    print('per POA alternations')
    print('total')
    print(100 * numpy.sum(analysis[:, :, 4], axis = 0) / numpy.sum(analysis[:, :, 0], axis = 0))
    #print('high Vowels')
    #print(100 * numpy.sum(analysis[:, :, 5], axis = 0) / numpy.sum(analysis[:, :, 1], axis = 0))
    #print('back Vowels')
    #print(100 * numpy.sum(analysis[:, :, 6], axis = 0) / numpy.sum(analysis[:, :, 2], axis = 0))
    print('per length alternations')
    print('total')
    print(100 * numpy.sum(analysis[:, :, 4], axis = 1) / numpy.sum(analysis[:, :, 0], axis = 1))
    #print('high Vowels')
    #print(100 * numpy.sum(analysis[:, :, 5], axis = 1) / numpy.sum(analysis[:, :, 1], axis = 1))
    #print('back Vowels')
    #print(100 * numpy.sum(analysis[:, :, 6], axis = 1) / numpy.sum(analysis[:, :, 2], axis = 1))
    print('per POA and length alternations')
    print(100 * analysis[:, :, 4] / analysis[:, :, 0])
    
    print('high Vowels per POA and length alternations')
    print(100 * analysis[:, :, 5] / analysis[:, :, 1])
    print('low Vowels per POA and length alternations')
    print(100 * (analysis[:, :, 4] - analysis[:, :, 5]) / (analysis[:, :, 0] - analysis[:, :, 1]))
    print('difference between high and low vowel alternations')
    print(100 * ((analysis[:, :, 5] / analysis[:, :, 1]) - (analysis[:, :, 4] - analysis[:, :, 5]) / (analysis[:, :, 0] - analysis[:, :, 1])))
    
    print('high Vowels per POA alternations')
    print(100 * numpy.sum(analysis[:, :, 5], axis = 0) / numpy.sum(analysis[:, :, 1], axis = 0))
    print('low Vowels per POA alternations')
    print(100 * numpy.sum(analysis[:, :, 4] - analysis[:, :, 5], axis = 0) / numpy.sum(analysis[:, :, 0] - analysis[:, :, 1], axis = 0))
    print('difference between high and low vowel alternations')
    print(100 * ((numpy.sum(analysis[:, :, 5], axis = 0) / numpy.sum(analysis[:, :, 1], axis = 0) - numpy.sum(analysis[:, :, 4] - analysis[:, :, 5], axis = 0) / numpy.sum(analysis[:, :, 0] - analysis[:, :, 1], axis = 0))))
    
    print('high Vowels alternations')
    print(100 * numpy.sum(analysis[:, :, 5]) / numpy.sum(analysis[:, :, 1]))
    print('low Vowels alternations')
    print(100 * numpy.sum(analysis[:, :, 4] - analysis[:, :, 5]) / numpy.sum(analysis[:, :, 0] - analysis[:, :, 1]))
    print('difference between high and low vowel alternations')
    print(100 * ((numpy.sum(analysis[:, :, 5]) / numpy.sum(analysis[:, :, 1])) - numpy.sum(analysis[:, :, 4] - analysis[:, :, 5]) / numpy.sum(analysis[:, :, 0] - analysis[:, :, 1])))
    
    print('back Vowels per POA and length alternations')
    print(100 * analysis[:, :, 6] / analysis[:, :, 2])
    print('back Vowels per POA and length alternations')
    print(100 * (analysis[:, :, 4] - analysis[:, :, 6]) / (analysis[:, :, 0] - analysis[:, :, 2]))
    print('difference between back and front vowel alternations')
    print(100 * ((analysis[:, :, 6] / analysis[:, :, 2]) - (analysis[:, :, 4] - analysis[:, :, 6]) / (analysis[:, :, 0] - analysis[:, :, 2])))
    
    print('back Vowels per POA alternations')
    print(100 * numpy.sum(analysis[:, :, 6], axis = 0) / numpy.sum(analysis[:, :, 2], axis = 0))
    print('front Vowels per POA alternations')
    print(100 * numpy.sum(analysis[:, :, 4] - analysis[:, :, 6], axis = 0) / numpy.sum(analysis[:, :, 0] - analysis[:, :, 2], axis = 0))
    print('difference between back and front vowel alternations')
    print(100 * ((numpy.sum(analysis[:, :, 6], axis = 0) / numpy.sum(analysis[:, :, 2], axis = 0) - numpy.sum(analysis[:, :, 4] - analysis[:, :, 6], axis = 0) / numpy.sum(analysis[:, :, 0] - analysis[:, :, 2], axis = 0))))
    
    print('back Vowels alternations')
    print(100 * numpy.sum(analysis[:, :, 6]) / numpy.sum(analysis[:, :, 2]))
    print('front Vowels alternations')
    print(100 * (numpy.sum(analysis[:, :, 4] - analysis[:, :, 6]) / numpy.sum(analysis[:, :, 0] - analysis[:, :, 2])))
    print('difference between back and front vowel alternations')
    print(100 * (numpy.sum(analysis[:, :, 6]) / numpy.sum(analysis[:, :, 2]) - numpy.sum(analysis[:, :, 4] - analysis[:, :, 6]) / numpy.sum(analysis[:, :, 0] - analysis[:, :, 2])))
    print('done')

if __name__ == "__main__":
    main()