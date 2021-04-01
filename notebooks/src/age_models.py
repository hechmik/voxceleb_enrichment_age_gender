import tensorflow
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

def compile_model(model, config):
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
            model.compile(loss=tensorflow.keras.losses.MeanAbsoluteError(),
                optimizer=opt,
                metrics=['mae'])
        elif config['loss'] == 'mse':
            model.compile(
                loss=tensorflow.keras.losses.MeanSquaredError(),
                optimizer=opt,
                metrics=['mae', 'mse'])
        elif config['loss'] == 'cross':
            model.compile(
                loss=tensorflow.keras.losses.CategoricalCrossentropy(),
                optimizer=opt,
                metrics=['categorical_accuracy', 'accuracy'])
        elif config['loss'] == 'cat_ordinal_loss':
            model.compile(
                loss=cat_ordinal_loss,
                optimizer=opt,
                metrics=['categorical_accuracy', 'accuracy'])
        elif config['loss'] in ['nll_poisson', 'nll_poisson_mse', 'nll_gaussian']:
            model.compile(
                loss=nll(loss_name=config['loss']),
                optimizer=opt,
                metrics=['mae', 'mse'])
        else:
            print('Invalid loss!')
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

def tdnn_model(input_shape, config):
    print(input_shape)
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(TDNNLayer(input_shape=(200*30,1)))
    model = add_dense_layers(model, config)
    compile_model(model, config)
    model.summary()
    return model

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

def pretrained_model(input_shape, config):
    if config['model_name'] == 'dense_net':
        model = DenseNet121(weights=None, input_shape=input_shape, include_top=False, pooling='avg')
    elif config['model_name'] == 'resnet':
        model = ResNet50(weights='imagenet', input_shape=input_shape, include_top=False, pooling='avg')
    elif config['model_name'] == 'resnet101':
        model = ResNet101V2(weights='imagenet', input_shape=input_shape, include_top=False, pooling='avg')
    from tensorflow.keras.models import Model
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

from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization,\
                                    Add, AveragePooling2D, Flatten, Dense, LeakyReLU, ELU
from tensorflow.keras.models import Model

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

def residual_block(x: Tensor, filters: int, relu_type: str, kernel_size: int, batch_norm: True ) -> Tensor:
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
    
    inputs = Input(shape=input_shape)
    num_filters = config['filter_n']
    t = inputs
    #t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size=config['kernel_size'],
               strides=1,
               filters=config['filter_n'],
               padding="same")(t)
    t = relu_bn(t, config['relu_type'], batch_norm=config['batch_norm'])
    if config['reduce_mel']:
        t = MaxPooling2D((config['pool_size'][0], 2))(t)
    else:
        t = MaxPooling2D((config['pool_size'][0], 1))(t)
    
    num_blocks_list = config['block_list']
    for i, num_blocks in enumerate(num_blocks_list):
        if i %2 == 1:
            t = MaxPooling2D(config['pool_size'])(t)
        for j in range(num_blocks):
            t = residual_block(t, filters=num_filters, kernel_size=config['kernel_size'], relu_type=config['relu_type'], batch_norm='batch_norm')
        num_filters *= 2
    if config['global_average']:
        t = GlobalAveragePooling2D()(t)
    else:
        t = AveragePooling2D((config['pool_size'][0] ** 2, config['pool_size'][1]))(t)
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
    
    model = Model(inputs, outputs)
    compile_model(model, config)
    model.summary()
    return model


def residual_block_1d(x: Tensor, filters: int, relu_type: str, kernel_size: int, batch_norm: True ) -> Tensor:
    y = Conv1D(kernel_size=kernel_size,
               strides= 1,
               filters=filters,
               padding="same")(x)
    y = relu_bn(y, relu_type=relu_type, batch_norm=batch_norm)
    y = Conv1D(kernel_size=kernel_size,
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

def create_res_net_1d(input_shape, config):
    
    inputs = Input(shape=input_shape)
    num_filters = config['filter_n']
    
    t = BatchNormalization()(inputs)
    t = Conv1D(kernel_size=config['kernel_size'],
               strides=1,
               filters=config['filter_n'],
               padding="same")(t)
    t = relu_bn(t, config['relu_type'], batch_norm=config['batch_norm'])
    
    num_blocks_list = config['block_list']
    for i, num_blocks in enumerate(num_blocks_list):
        t = MaxPooling1D(config['pool_size'])(t)
        for j in range(num_blocks):
            t = residual_block_1d(
                t,
                filters=num_filters,
                kernel_size=config['kernel_size'],
                relu_type=config['relu_type'],
                batch_norm='batch_norm')
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
    return model