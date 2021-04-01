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
from src.tdnn import TDNNLayer
from src.utils_models import compile_model


def lasso_custom(input_shape, config):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))    
    model.add(Dense(1,
                    activation='linear',
                    kernel_initializer=config['kernel_initializer'],
                    kernel_regularizer=l1(config['l_reg'])))
    compile_model(model, config)
    model.summary()
    return model

def fc_custom(input_shape, config):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(config['dense_n'],
                    activation='relu',
                    kernel_initializer=config['kernel_initializer'],
                    kernel_regularizer=l1(config['l_reg'])))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    for i in range(config['extra_dense_layers']):
        model.add(Dense(config['dense_n'],
                    activation='relu',
                    kernel_initializer=config['kernel_initializer'],
                    kernel_regularizer=l1(config['l_reg'])))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))                   
    model.add(Dense(1,
                    activation='linear',
                    kernel_initializer=config['kernel_initializer'],
                    kernel_regularizer=l1(config['l_reg'])))
    compile_model(model, config)
    model.summary()
    return model