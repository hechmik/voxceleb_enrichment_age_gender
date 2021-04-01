import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import tensorflow
tensorflow.config.threading.set_intra_op_parallelism_threads(4)
tensorflow.config.threading.set_inter_op_parallelism_threads(2)
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, BatchNormalization, Dense, Dropout, Input, LSTM, GlobalAveragePooling1D, GlobalAveragePooling2D, AveragePooling1D
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import DenseNet121,ResNet50, ResNet101V2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l2, l1
from tensorflow.python.keras import backend as K
import sys
sys.path.insert(1, '.')
from src.utils_models import compile_model, add_dense_layers
from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization,\
                                    Add, AveragePooling2D, Flatten, Dense, LeakyReLU, ELU
from tensorflow.keras.models import Model

def cnn_two_conv_one_max(input_shape, apply_dropout, config, num_blocks=1):
    
    model = Sequential()
    for i in range(0, num_blocks):
        
        model.add(Conv2D(2 ** (2*i) * config['filter_n'],
                         kernel_size=config['kernel_size'],
                         activation='relu',
                         padding='same',
                         input_shape=input_shape,
                         kernel_initializer=config['kernel_initializer'],
                         kernel_regularizer=l2(config['l_reg'])
                        ))
        model.add(Conv2D(2 ** (2*i +1) * config['filter_n'],
                         kernel_size=config['kernel_size'],
                         activation='relu',
                         padding='same',
                         kernel_initializer=config['kernel_initializer'],
                         kernel_regularizer=l2(config['l_reg'])
                         ))
        if config['batch_norm_everywhere']:
            model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(config['pool_size'])))
    
    model.add(Flatten())
    model = add_dense_layers(model, config)
    compile_model(model, config)
    model.summary()
    return model


def cnn_custom_1d(input_shape, apply_dropout, config):
    
    model = Sequential()
    model.add(Conv1D(config['filter_n'],
                     kernel_size=config['kernel_size'],
                     activation='relu',
                     padding='same',
                     strides=config['strides'],
                     input_shape=input_shape,
                     kernel_initializer=config['kernel_initializer'],
                     kernel_regularizer=l2(config['l_reg'])
                    ))
    if config['batch_norm_everywhere']:
        model.add(BatchNormalization())
    model.add(Conv1D(2 * config['filter_n'],
                     kernel_size=config['kernel_size'],
                     activation='relu',
                     padding='same',
                     strides=config['strides'],
                     kernel_initializer=config['kernel_initializer'],
                     kernel_regularizer=l2(config['l_reg'])
                    ))
    if config['batch_norm_everywhere']:
        model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(config['pool_size'][0])))
    
    model.add(Conv1D(4 * config['filter_n'],
                     kernel_size=config['kernel_size'],
                     activation='relu',
                     padding='same',
                     kernel_initializer=config['kernel_initializer'],
                     kernel_regularizer=l2(config['l_reg'])
                    ))
    if config['batch_norm_everywhere']:
        model.add(BatchNormalization())
    model.add(Conv1D(8 * config['filter_n'],
                     kernel_size=config['kernel_size'],
                     activation='relu',
                     padding='same',
                     kernel_initializer=config['kernel_initializer'],
                     kernel_regularizer=l2(config['l_reg'])
                    ))
    if config['batch_norm_everywhere']:
        model.add(BatchNormalization())
    if config['global_average']:
        model.add(GlobalAveragePooling1D())
    else:
        model.add(MaxPooling1D(pool_size=(config['pool_size'][0])))
    model.add(Flatten())
    model = add_dense_layers(model, config)

    compile_model(model, config)
    model.summary()
    return model


def cnn_custom(input_shape, apply_dropout, config):
    model = cnn_two_conv_one_max(input_shape, apply_dropout, config, 2)
    return model

def cnn_alternating_blocks(input_shape, apply_dropout, config, num_blocks):
    model = Sequential()
    model.add(Conv2D(config['filter_n'],
                     kernel_size=config['kernel_size'],
                     activation='relu',
                     padding='same',
                     input_shape=input_shape,
                     kernel_initializer=config['kernel_initializer'],
                     kernel_regularizer=l2(config['l_reg']),
                     bias_regularizer=l2(config['l_reg'])
                    ))
    if config['batch_norm_everywhere']:
        model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(config['pool_size'])))
    
    for i in range(1, num_blocks): 
        model.add(Conv2D(2 ** i * config['filter_n'],
                         kernel_size=config['kernel_size'],
                         activation='relu',
                         padding='same',
                         kernel_initializer=config['kernel_initializer'],
                         kernel_regularizer=l2(config['l_reg']),
                         bias_regularizer=l2(config['l_reg'])))
        if config['batch_norm_everywhere']:
            model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(config['pool_size'])))
    model.add(Flatten())
    model = add_dense_layers(model, config)
    compile_model(model, config)
    model.summary()
    return model

def cnn_three_blocks(input_shape, apply_dropout, config):
    model = cnn_alternating_blocks(input_shape, apply_dropout, config, 3)
    return model


def cnn_four_blocks(input_shape, apply_dropout, config):
    model = cnn_alternating_blocks(input_shape, apply_dropout, config, 4)
    return model