from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import activations
import numpy as np


class TDNNLayer(Layer):
    """TDNNLayer
    TDNNLayer sounds like 1D conv with extra steps. Why not doing it with Keras ?
    This layer inherits the Layer class from Keras and is inspired by conv1D layer.
    This is a simple Keras layer. It is called like a Conv2D layer with some more parameters :
    TDNNLayer(input_context, sub_sampling, initializer, activation)
        input_context: the size of the window [-2,2] for example (-2 index and 2 index)
        sub_sampling: true to keep only extreme index (default to False)
        initializer: weight initializer (default to uniform)
        activation: activation function to use (default to none)
    Example to add a TDNN layer to your model :
        from keras.models import Sequential
        model = Sequential()
        model.add(TDNNLayer())
    """

    def __init__(self,
                 input_context=[-2, 2],
                 sub_sampling=False,
                 initializer='uniform',
                 activation=None,
                 **kwargs):

        self.input_context = input_context
        self.sub_sampling = sub_sampling
        self.initializer = initializer
        self.activation = activations.get(activation)
        self.mask = None
        self.kernel = None
        super(TDNNLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        kernel_shape = (1, self.input_context[1]-self.input_context[0]+1, 1, 1)
        self.kernel = self.add_weight(name='kernel',
                                      shape=kernel_shape,
                                      initializer=self.initializer,
                                      trainable=True)
        self.mask = np.zeros(kernel_shape)
        self.mask[0][0][0] = 1
        self.mask[self.input_context[1]-self.input_context[0]][0][0] = 1
        print(self.mask.shape)

        if self.sub_sampling:
            self.kernel = self.kernel * self.mask
        print(self.kernel.shape)

        super(TDNNLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self,
             inputs,
             mask=None,
             training=None,
             initial_state=None,
             constants=None):
        if self.sub_sampling:
            output = K.conv1d(inputs,
                              self.kernel,
                              strides=1,
                              padding="same",
                              )
        else:
            masked_kernel = self.kernel * self.mask
            output = K.conv1d(inputs,
                              masked_kernel,
                              strides=1,
                              padding="same",
                              )
        if self.activation is not None:
            return self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1]-self.input_context[1]+self.input_context[0]