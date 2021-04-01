import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import tensorflow
tensorflow.config.threading.set_intra_op_parallelism_threads(2)
tensorflow.config.threading.set_inter_op_parallelism_threads(1)
from tensorflow.keras.layers import Flatten, BatchNormalization, Dense, Dropout, Input
from tensorflow.keras import Sequential
from tensorflow import Tensor
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.keras import backend as K
from tensorflow.keras.layers import ReLU, BatchNormalization, LeakyReLU, ELU
import sys
from src.tdnn import TDNNLayer
import time

def compile_model(model, config):
    print("Compile_model >>>")
    """
    time.sleep(15)"""
    if config['optimizer'] == 'adam':
        if config['model_name'] in ['lstm', 'vanilla_lstm']:
            opt = Adam(learning_rate=config['lr'], clipvalue=0.5)
        else:
            opt = Adam(learning_rate=config['lr'])
    elif config['optimizer'] == 'sgd':
        if config['decay_rate']:
            decay_rate = config['lr'] / config['epochs']
            opt = SGD(lr=config['lr'], decay=decay_rate, momentum=config['momentum'])
        else:
            opt = SGD(lr=config['lr'], momentum=config['momentum'])
    elif config['optimizer'] == 'padam':
        from src.padam import Padam
        opt = Padam(lr=config['lr'], partial=0.125)
    if config['multi_output']:
        model.compile(
            optimizer=opt, 
            loss={
                'regression': 'mse', 
                'classifier': 'categorical_crossentropy'
            },
            metrics={
                'regression': 'mae', 
                'classifier': 'accuracy'
            },
            loss_weights={
                'regression': 0.1,
                'classifier': 10
            }
        )
    else:
        if config['loss']=='mae':
            loss=tensorflow.keras.losses.MeanAbsoluteError()
            metrics=['mae']
        elif config['loss'] == 'mse':
            loss=tensorflow.keras.losses.MeanSquaredError()
            metrics=['mae', 'mse']
        elif config['loss'] == 'cross':
            loss=tensorflow.keras.losses.CategoricalCrossentropy()
            metrics=['categorical_accuracy', 'accuracy']
        elif config['loss'] == 'cat_ordinal_loss':
            loss=cat_ordinal_loss
            metrics=['categorical_accuracy', 'accuracy']
        elif config['loss'] in ['nll_poisson', 'nll_poisson_mse', 'nll_gaussian']:
            loss=nll(loss_name=config['loss'])                
            metrics=['mae', 'mse']
        else:
            print('Invalid loss!')
        model.compile(
                loss=loss,
                optimizer=opt,
                metrics=metrics
            )
        """
        print("compile_model <<<")
        time.sleep(15)
        """
def nll(loss_name):
    
    from tensorflow_probability.python.distributions import Categorical, Poisson, NegativeBinomial, Gamma, LogNormal
    def nll_loss(y_true, y_pred):
    # for Poisson n_outputs=1 (regression task)
        diff = tensorflow.math.abs(y_true - y_pred)
        if loss_name in ['nll_poisson', 'nll_poisson_mse']:
            y_true_dist = Poisson(rate=y_true)
            loss = -y_true_dist.log_prob(y_pred)
        elif loss_name == 'nll_gaussian':
            loss = 0
        else:
            loss = 0
        if loss_name == 'nll_poisson_mse':
            loss = tensorflow.math.multiply(diff, loss)
        return K.mean(loss, axis=-1)
    print(nll_loss)
    return nll_loss

def cat_ordinal_loss(y_true, y_pred):
    # Source https://github.com/JHart96/keras_ordinal_categorical_crossentropy/blob/master/ordinal_categorical_crossentropy.py
    weights = K.cast(K.abs(K.argmax(y_true, axis=1) - K.argmax(y_pred, axis=1))/(K.int_shape(y_pred)[1] - 1), dtype='float32')
    return (1.0 + weights) * tensorflow.keras.losses.categorical_crossentropy(y_true, y_pred)

def custom_loss_mse(y_true, y_pred, threshold=5, power=3):
    diff = tensorflow.math.abs(y_true - y_pred)
    loss = K.switch(K.greater(diff, threshold), K.pow(diff, power), K.pow(diff, 2))
    return K.mean(loss, axis=-1)

def add_dense_layers(model, config):
    if config['initial_dropout']:
        model.add(Dropout(0.2))
    model.add(Dense(config['dense_n'],
                    activation='relu',
                    kernel_initializer=config['kernel_initializer'],
                    kernel_regularizer=l2(config['l_reg'])))
    # Order of batch norm and dropout
    if config['batch_norm_everywhere']:
        model.add(BatchNormalization())
    if config['dropout']:
        model.add(Dropout(0.5))
    if config['2nd_dense_n'] > 0:
        model.add(Dense(config['2nd_dense_n'],
                        activation='relu',
                        kernel_initializer=config['kernel_initializer'],
                        kernel_regularizer=l2(config['l_reg'])))
        if config['batch_norm_everywhere']:
            model.add(BatchNormalization())
        if config['dropout']:
            model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer=config['kernel_initializer']))
    return model

def relu_bn(inputs: Tensor, relu_type: str, batch_norm) -> Tensor:
    if relu_type == 'leaky_relu':
        relu = LeakyReLU(alpha=0.1)(inputs)
    elif relu_type == 'elu':
        relu = ELU()(inputs)
    else:
        relu = ReLU()(inputs)
    if batch_norm:
        bn = BatchNormalization()(relu)
    else:
        bn = relu
    return bn
