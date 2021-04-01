from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, BatchNormalization, Dense, Dropout, Input, LSTM, GlobalAveragePooling1D, GlobalAveragePooling2D, AveragePooling1D
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import DenseNet121,ResNet50, ResNet101V2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l2, l1
from tensorflow.python.keras import backend as K
import sys
sys.path.insert(1, '.')
from src.utils_models import compile_model
from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization,\
                                    Add, AveragePooling2D, Flatten, Dense, LeakyReLU, ELU
from tensorflow.keras.models import Model
import tensorflow

def lstm_cnn(input_shape, config):
    model = Sequential()
    model.add(LSTM(
        config['lstm_cells'],
        return_sequences=True,
        input_shape=input_shape,
        kernel_initializer=config['kernel_initializer']
    ))
    model.add(LSTM(
        config['lstm_cells'],
        return_sequences=True,
        kernel_initializer=config['kernel_initializer']
    ))
    
    model.add(Conv1D(config['filter_n'],
                     kernel_size=config['kernel_size'],
                     activation='relu',
                     padding='same',
                     strides=config['strides'],
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
    model.add(Flatten())
    model.add(Dense(
        config['dense_n'],
        kernel_initializer=config['kernel_initializer'],
        activation="relu"
    ))
    model.add(Dense(
        1,
        kernel_initializer=config['kernel_initializer']
    ))    
    compile_model(model, config)
    model.summary()
    return model


def cnn_lstm(input_shape, config):
    model = Sequential()
    
    
    model.add(Conv1D(config['filter_n'],
                     kernel_size=config['kernel_size'],
                     activation='relu',
                     padding='same',
                     strides=config['strides'],
                     kernel_initializer=config['kernel_initializer'],
                     kernel_regularizer=l2(config['l_reg']),
                     input_shape=input_shape
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
    
    model.add(LSTM(
        config['lstm_cells'],
        return_sequences=True,
        input_shape=input_shape,
        kernel_initializer=config['kernel_initializer']
    ))
    model.add(LSTM(
        config['lstm_cells'],
        return_sequences=False,
        kernel_initializer=config['kernel_initializer']
    ))
    
    model.add(Flatten())
    model.add(Dense(
        config['dense_n'],
        kernel_initializer=config['kernel_initializer'],
        activation="relu"
    ))
    model.add(Dense(
        1,
        kernel_initializer=config['kernel_initializer']
    ))    
    compile_model(model, config)
    model.summary()
    return model

