'''
Created on Aug 26, 2016

@author: veltzer
'''
# -*- coding: utf-8 -*-

from keras import backend as K
from keras.initializations import identity
from keras.layers import SimpleRNN, InputSpec

class MineRNN(SimpleRNN):
    '''RNN with a dot product based recursive connection where only the neuron's own output is fed back to its input via the hidden state.
    # Arguments
        same as SimpleRNN
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: this is disregarded, instead ones(output_dim) is used.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.
    
    '''
    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]
        input_dim = input_shape[2]
        self.input_dim = input_dim
        
        self.W = self.init((input_dim, self.output_dim),
                           name='{}_W'.format(self.name))
        self.U = identity((self.output_dim, self.output_dim),
                        name='{}_U'.format(self.name))
        self.b = K.zeros((self.output_dim,),
                         name='{}_b'.format(self.name))
        
        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)
        
        self.trainable_weights = [self.W, self.b]
        
        self.non_trainable_weights = [self.U]
        
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights