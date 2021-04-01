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
from src.utils_models import compile_model, relu_bn
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization,\
                                    Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow import Tensor
import time


def residual_block(x: Tensor, filters: int, relu_type: str, kernel_size: int, batch_norm: True, fixed_resnet: False) -> Tensor:
    if fixed_resnet:
        x = Conv2D(kernel_size=kernel_size,
               strides= (1),
               filters=filters,
               padding="same")(x)
        x = relu_bn(x, relu_type=relu_type, batch_norm=batch_norm)
        y = Conv2D(kernel_size=kernel_size,
                   strides=1,
                   filters=filters,
                   padding="same")(x)
        if batch_norm:
            y = BatchNormalization()(y)
        out = Add()([x, y])
    else:    
        y = Conv2D(kernel_size=kernel_size,
                   strides= (1),
                   filters=filters,
                   padding="same")(x)
        y = relu_bn(y, relu_type=relu_type, batch_norm=batch_norm)
        y = Conv2D(kernel_size=kernel_size,
                   strides=1,
                   filters=filters,
                   padding="same")(y)
        if batch_norm:
            y = BatchNormalization()(y)
        if x.shape[-1] == y.shape[-1]:
            out = Add()([x, y])
        else:
            out = y
    out = relu_bn(out, relu_type=relu_type, batch_norm=batch_norm)
    return out

def create_res_net(input_shape, config):
    print("create_res_net >>>")
    inputs = Input(shape=input_shape)
    num_filters = config['filter_n']
    
    t=inputs
    #t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size=config['kernel_size'],
               strides=1,
               filters=config['filter_n'],
               padding="same")(t)

    t = relu_bn(t, config['relu_type'], batch_norm=config['batch_norm'])
    if config['reduce_mel']:
        t = MaxPooling2D((config['pool_size'][0], 2))(t)
    else:
        t = MaxPooling2D(config['pool_size'])(t)
    
    num_blocks_list = config['block_list']

    for i, num_blocks in enumerate(num_blocks_list):
        if i > 0:
            t = MaxPooling2D(config['pool_size'])(t)
        for j in range(num_blocks):
            t = residual_block(t, filters=num_filters, kernel_size=config['kernel_size'], relu_type=config['relu_type'], batch_norm='batch_norm')
        num_filters *= 2
    if config['global_average']:
        t = GlobalAveragePooling2D()(t)
    else:
        t = AveragePooling2D(config['pool_size'])(t)
    t = Flatten()(t)
    t = Dropout(0.2)(t)
    t = Dense(config['dense_n'])(t)
    t = relu_bn(t, config['relu_type'], batch_norm=config['batch_norm'])
    t = Dropout(0.5)(t)


    if config['2nd_dense_n'] > 0:
        t = Dense(config['2nd_dense_n'])(t)
        t = relu_bn(t, config['relu_type'], batch_norm=config['batch_norm'])
        t = Dropout(0.5)(t)
    outputs = []
    classifier = Dense(config['n_categories'], activation='softmax', name='classifier')(t)
    regression = Dense(1, activation='linear', name='regression')(t)
    if config['n_categories'] > 0 and config['multi_output']:
        outputs.append(regression)
        outputs.append(classifier)
    elif config['n_categories'] > 0:
        outputs.append(classifier)
    else:
        outputs.append(regression)
    """
    print("Before model init")
    time.sleep(10)"""
    model = Model(inputs, outputs)
    compile_model(model, config)
    model.summary()
    return model


def residual_block_1d(x: Tensor, filters: int, relu_type: str, kernel_size: int, batch_norm, dilation_rate:int, strides:int, fixed_resnet: bool ) -> Tensor:
    if fixed_resnet:
        x = Conv1D(kernel_size=kernel_size,
               filters=filters,
               padding="same",
               dilation_rate=dilation_rate,
               strides=strides)(x)
        x = relu_bn(x, relu_type=relu_type, batch_norm=batch_norm)
        y = Conv1D(kernel_size=kernel_size,
                   filters=filters,
                   padding="same",
                   dilation_rate=dilation_rate,
                   strides=strides)(x)
        if batch_norm:
            y = BatchNormalization()(y)
        out = Add()([x, y])
    else:
        y = Conv1D(kernel_size=kernel_size,
                   filters=filters,
                   padding="same",
                   dilation_rate=dilation_rate,
                   strides=strides)(x)
        y = relu_bn(y, relu_type=relu_type, batch_norm=batch_norm)
        y = Conv1D(kernel_size=kernel_size,
                   filters=filters,
                   padding="same",
                   dilation_rate=dilation_rate,
                   strides=strides)(y)
        if batch_norm:
            y = BatchNormalization()(y)
        if x.shape[-1] == y.shape[-1]:
            out = Add()([x, y])
        else:
            out = y
    out = relu_bn(out, relu_type=relu_type, batch_norm=batch_norm)
    return out

def create_res_net_1d(input_shape, config):
    print("create_res_net_1d >>>")
    inputs = Input(shape=input_shape)
    num_filters = config['filter_n']
    t = inputs
    if not config['without_initial_batch_norm']:
        t = BatchNormalization()(t)
    dilation_rate=1
    strides = 1
    if 'dilation_rate' in config.keys():
        dilation_rate=config['dilation_rate']
        strides = config['strides']
    fixed_resnet = False
    if 'fixed_resnet' in config.keys():
        fixed_resnet=config['fixed_resnet']
    t = Conv1D(kernel_size=config['kernel_size'],
               filters=num_filters,
               padding="same",
               dilation_rate=dilation_rate,
               strides=strides
              )(t)
    t = relu_bn(t, config['relu_type'], batch_norm=config['batch_norm'])
    
    num_blocks_list = config['block_list']
    for i, num_blocks in enumerate(num_blocks_list):
        if i>0:
            t = MaxPooling1D(config['pool_size'])(t)
        for j in range(num_blocks):
            t = residual_block_1d(
                t,
                filters=num_filters,
                kernel_size=config['kernel_size'],
                relu_type=config['relu_type'],
                batch_norm=config['batch_norm'],
                dilation_rate=dilation_rate,
                strides=strides,
                fixed_resnet=fixed_resnet
            )
        if 'decreasing_filters' in config.keys():
            if num_filters > 3:
                num_filters = num_filters / 2
        else:
            num_filters *= 2
    if config['global_average']:
        t = GlobalAveragePooling1D()(t)
    else:
        t = AveragePooling1D(config['pool_size'] ** 2)(t)
    t = Flatten()(t)
    t = Dropout(0.2)(t)
    t = Dense(config['dense_n'])(t)
    t = relu_bn(t, config['relu_type'], batch_norm=config['batch_norm'])
    t = Dropout(0.5)(t)


    if config['2nd_dense_n'] > 0:
        t = Dense(config['2nd_dense_n'])(t)
        t = relu_bn(t, config['relu_type'], batch_norm=config['batch_norm'])
        t = Dropout(0.5)(t)
    outputs = []
    
    regression = Dense(1, activation='linear', name='regression')(t)
    if config['n_categories'] > 0:
        classifier = Dense(config['n_categories'], activation='softmax', name='classifier')(t)
        if config['multi_output']:
            outputs.append(regression)
        outputs.append(classifier)
    else:
        outputs.append(regression)
    
    model = Model(inputs, outputs)
    compile_model(model, config)
    model.summary()
    print("create_res_net_1d <<<")
    return model
