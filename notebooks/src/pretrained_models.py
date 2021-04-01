import tensorflow
from tensorflow.keras.layers import Flatten, BatchNormalization, Dense, Dropout, Input, GlobalAveragePooling1D, GlobalAveragePooling2D, AveragePooling1D
from tensorflow.keras import Sequential
from tensorflow.keras.applications import DenseNet121,ResNet50, ResNet101V2
from tensorflow.keras.regularizers import l2, l1
from tensorflow.python.keras import backend as K
import sys
sys.path.insert(1, '.')
from src.tdnn import TDNNLayer
from src.utils_models import compile_model, relu_bn
from tensorflow import Tensor


def pretrained_model(input_shape, config):
    if config['model_name'] == 'dense_net':
        model = DenseNet121(weights=None, input_shape=input_shape, include_top=False, pooling='avg')
    elif config['model_name'] == 'resnet50':
        model = ResNet50(weights='imagenet', input_shape=input_shape, include_top=False, pooling='avg')
    elif config['model_name'] == 'resnet101':
        model = ResNet101V2(weights='imagenet', input_shape=input_shape, include_top=False, pooling='avg')
    from tensorflow.keras.models import Model
    if config['train_only_fc']:
        model.trainable = False
    inputs = Input(input_shape)
    t = model(inputs)
    t = Dropout(0.2)(t)
    t = Dense(config['dense_n'])(t)
    t = relu_bn(t, config['relu_type'], batch_norm=config['batch_norm'])
    t = Dropout(0.5)(t)


    if config['2nd_dense_n'] > 0:
        t = Dense(config['2nd_dense_n'])(t)
        t = relu_bn(t, config['relu_type'], batch_norm=config['batch_norm'])
        t = Dropout(0.5)(t)
    out = Dense(1, activation='linear')(t)
    model = Model(inputs=inputs,outputs=out)
    compile_model(model, config)
    model.summary()
    return model