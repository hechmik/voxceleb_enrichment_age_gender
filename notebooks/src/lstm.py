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

def lstm_model(input_shape, config):
    model = Sequential()
    model.add(LSTM(
        config['lstm_cells'],
        return_sequences=True,
        input_shape=input_shape,
        kernel_initializer=config['kernel_initializer']
    ))
    model.add(LSTM(
        config['lstm_cells'],
        kernel_initializer=config['kernel_initializer']
    ))
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

def vanilla_lstm_model(input_shape, config):
    
    model = Sequential()
    model.add(LSTM(
        config['lstm_cells'],
        input_shape=input_shape,
        kernel_initializer=config['kernel_initializer']
    ))
    model.add(Dense(
        config['dense_n'],
        kernel_initializer=config['kernel_initializer'],
        activation="relu"
    ))
    model.add(Dense(
        1,
        kernel_initializer=config['kernel_initializer'],
        activation='linear'
    ))   
    compile_model(model, config)
    model.summary()
    return model