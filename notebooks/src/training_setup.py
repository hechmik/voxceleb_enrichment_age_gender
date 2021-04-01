import json
import gc
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import numpy as np
from numpy import empty
import sklearn.model_selection
import scipy.fftpack
from tqdm import tqdm
import itertools
import wandb
import wandb.keras
import sklearn.model_selection
from wandb.keras import WandbCallback
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tensorflow
import sklearn.model_selection
import sys
sys.path.insert(1, '.')
from src.cnn import cnn_custom, cnn_four_blocks, cnn_three_blocks, cnn_two_conv_one_max, cnn_custom_1d
from src.lstm import lstm_model, vanilla_lstm_model
from src.resnets import create_res_net, create_res_net_1d
from src.pretrained_models import pretrained_model
from src.combined_nn import lstm_cnn, cnn_lstm
from src.dense_models import lasso_custom, fc_custom
from src.age_models import tdnn_model
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)
import time

class GCCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        #gc.collect()
        import resource
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_col, y_col, feat_len, preprocessing_strategy, preprocess_values, model_name, resnet_mode,
                 batch_size=32, shuffle=True, shuffle_temporal=False, multi_output=None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.x_col = x_col
        self.y_col = y_col
        self.feat_len = feat_len
        self.pre_strat = preprocessing_strategy
        self.pre_values = preprocess_values
        self.indices = np.arange(self.x_col.shape[0])
        self.resnet_mode = resnet_mode
        self.shuffle_temporal = shuffle_temporal
        self.model_name = model_name
        self.multi_output = multi_output

    def __len__(self):
        return int(np.ceil(len(self.x_col) / float(self.batch_size)))

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.multi_output:
            y_0 = np.array(self.y_col[0])[inds]
            y_1 = self.y_col[1][inds, :]
            y = [y_0, y_1]
            y_0 = None
            y_1 = None
            del y_0, y_1
        else:
            y = self.y_col[inds]
        x = self.x_col[inds]
        # read your data here using the batch lists, batch_x and batch_y
        x = [self.pick_recs(mfcc_batch) for mfcc_batch in x]
        x = np.stack(x)
        if self.model_name not in ['lstm', 'vanilla_lstm', 'cnn_custom_1d', 'lstm_cnn', 'tdnn_model', 'cnn_resnet_1d']:
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
        if self.resnet_mode:
            x = np.repeat(x, 3, axis=3)
        if self.pre_strat in ['norm_mfcc', 'min_max']:
            if len(x.shape) < len(self.pre_values[0].shape):
                first_pre_val = self.pre_values[0].reshape(self.pre_values[0].shape[0],
                                                           self.pre_values[0].shape[1],
                                                           self.pre_values[0].shape[2])

                second_pre_val = self.pre_values[1].reshape(self.pre_values[1].shape[0],
                                                            self.pre_values[1].shape[1],
                                                            self.pre_values[1].shape[2])
                self.pre_values = (first_pre_val, second_pre_val)
            if self.pre_strat == 'norm_mfcc':
                mean, std = self.pre_values
                x = (x - mean) / std
            elif self.pre_strat == 'min_max':
                max_x, min_x = self.pre_values
                x = (x - min_x) / (max_x - min_x)
        return x, y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def pick_recs(self, batch):
        if batch.shape[0] > self.feat_len:
            start_point = np.random.randint(0, batch.shape[0] - self.feat_len + 1)
            end_point = start_point + self.feat_len
            batch = batch[start_point:end_point, :]
        if self.shuffle_temporal:
            idx = list(range(batch.shape[0]))
            np.random.shuffle(idx)
            batch = batch[idx, :]
        if self.pre_strat == 'norm_mfcc_dataloader':
            mean = np.mean(batch, axis=0)
            std = np.std(batch, axis=0)
            epsilon= 0.001 ** 10
            std = np.array([x if x > 0 else epsilon for x in std])
            batch = (batch - mean) / std
        elif self.pre_strat == 'sub_mean_dataloader':
            batch = batch - np.mean(batch, axis=0)
        return batch

import sys
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


def kfold_cv(X, y, X_spk_labels, X_spk_labels_aug, X_aug, y_aug, preprocessing_strategy, model_name, config):

    print('kfold_cv >>>')
    kfold = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=config['seed'])
    fold_iteration = 0
    maes = []
    std_maes = []
    if config['unbalanced']:
        idx_split = np.unique(X_spk_labels)
    else:
        idx_split = y
    for train_index, test_index in kfold.split(idx_split):
        train_embeddings = X[train_index,]
        test_embeddings = X[test_index,]
        train_labels = y[train_index]
        test_labels = y[test_index]
        if config['unbalanced'] and config['data_augmentation']:
            train_spk = idx_split[train_index]
            test_spk = idx_split[test_index]
            train_idx = []
            test_idx = {}
            # Find Train and Test obs
            for i, spk in enumerate(X_spk_labels):
                if spk in train_spk:
                    train_idx.append(i)
                elif spk in test_spk:
                    test_idx[spk] = i
            train_embeddings = X[train_idx,]
            test_idx = list(test_idx.values())
            test_embeddings = X[test_idx,]
            train_labels = y[train_idx]
            test_labels = y[test_idx]
            # Now data augmentation
            train_aug_idx = []
            for i, spk in enumerate(X_spk_labels_aug):
                if spk in train_spk:
                    train_aug_idx.append(i)
            print("Shape before:", train_embeddings.shape)
            train_embeddings = np.concatenate((train_embeddings, X_aug[train_aug_idx,]))
            print("Shape after:", train_embeddings.shape)
            train_labels = np.concatenate((train_labels, y_aug[train_aug_idx]))

        elif config['data_augmentation']:
            train_spk = X_spk_labels[train_index]
            train_aug_idx = []
            spk_aug = []
            low_bound = np.percentile(train_labels, 25)
            upp_bound = np.percentile(train_labels, 75)
            print('Low_bound', low_bound, 'UP_bound', upp_bound)
            # Find the idx, in the augmented dataset, of speakers that are in the current train fold
            for i, spk in enumerate(X_spk_labels_aug):
                if spk in train_spk:
                    if config['selective_data_aug']:
                        if y_aug[i] < low_bound or y_aug[i] > upp_bound:
                            train_aug_idx.append(i)
                            spk_aug.append(spk)
                    else:
                        train_aug_idx.append(i)
                        spk_aug.append(spk)
            print("Shape before:", train_embeddings.shape)
            train_embeddings = np.concatenate((train_embeddings, X_aug[train_aug_idx,]))
            print("Shape after:", train_embeddings.shape)
            train_labels = np.concatenate((train_labels, y_aug[train_aug_idx]))

        if config['include_title_only_obs']:
            if config['unbalanced_include_title_only_obs']:
                print("Loading unbalanced_include_title_only_obs")
                data = np.load('numpy_train_test_obj_unbalanced-title_only.npz', allow_pickle=True)
            else:
                data = np.load('numpy_train_test_obj-title_only.npz', allow_pickle=True)

            vectors = []
            for x in list(data.keys()):
                vectors.append(data[x])

            X_title_only, y_title_only, X_spk_labels_title_only, X_spk_labels_aug_title_only, X_aug_title_only, y_aug_title_only = vectors
            if config['embedding'] == 'mel_spect_kaldi':
                X_title_only_mel = empty(X_title_only.shape, dtype='object')
                for i in range(X_title_only.shape[0]):
                    temp = scipy.fftpack.idct(X_title_only[i])
                    X_title_only_mel[i] = temp
                X_title_only = X_title_only_mel
                X_aug_title_only_mel = empty(X_aug_title_only.shape, dtype='object')
                for i in range(X_aug_title_only.shape[0]):
                    X_aug_title_only_mel[i] = scipy.fftpack.idct(X_aug_title_only[i])
                X_aug_title_only = X_aug_title_only_mel
            print("Shape before include_title_only_obs:", train_embeddings.shape)
            train_embeddings = np.concatenate((train_embeddings, X_title_only))
            train_embeddings = np.concatenate((train_embeddings, X_aug_title_only))
            print("Shape after include_title_only_obs:", train_embeddings.shape)
            train_labels = np.concatenate((train_labels, y_title_only))
            train_labels = np.concatenate((train_labels, y_aug_title_only))
            data.close()
            data = None
            del data
            del X_title_only, y_title_only, X_spk_labels_title_only, X_spk_labels_aug_title_only, X_aug_title_only, y_aug_title_only, vectors
            import gc
            gc.collect()

        if config['sampling_strategy'] == 'undersampling':
            train_embeddings, train_labels = undersample_dataset(config, train_embeddings, train_labels)
        elif config['sampling_strategy'] == 'oversampling':
            train_embeddings, train_labels = oversample_dataset(train_embeddings, train_labels)
            
        scaler = None
        preprocess_values = None
        print("Preprocess strat:", preprocessing_strategy)
        if preprocessing_strategy == 'norm_mfcc':
            mean = np.array([np.mean(x, axis=0) for x in train_embeddings])
            mean = np.mean(mean, axis=0)
            mean = mean.reshape(1, 1, mean.shape[0], 1)

            std = np.array([np.std(x, axis=0) for x in train_embeddings])
            std = np.mean(std, axis=0)
            std = std.reshape(1, 1, std.shape[0], 1)
            preprocess_values = (mean, std)
        elif preprocessing_strategy == 'min_max':
            max_x = np.array([np.max(x, axis=0) for x in train_embeddings])
            max_x = np.max(max_x, axis=0)
            max_x = max_x.reshape(1, 1, max_x.shape[0], 1)

            min_x = np.array([np.min(x, axis=0) for x in train_embeddings])
            min_x = np.min(min_x, axis=0)
            min_x = min_x.reshape(1, 1, min_x.shape[0], 1)
            preprocess_values = (max_x, min_x)
        elif preprocessing_strategy == 'sub_mean':
            means = [np.mean(x, axis=0) for x in train_embeddings]
            for i in range(train_embeddings.shape[0]):
                train_embeddings[i] = train_embeddings[i] - means[i]
            means = [np.mean(x, axis=0) for x in test_embeddings]
            for i in range(test_embeddings.shape[0]):
                test_embeddings[i] = test_embeddings[i] - means[i]
        elif preprocessing_strategy == 'sub_mean_div_std':
            means = [np.mean(x, axis=0) for x in train_embeddings]
            std = [np.std(x, axis=0) for x in train_embeddings]
            for i in range(train_embeddings.shape[0]):
                train_embeddings[i] = (train_embeddings[i] - means[i]) / std[i]
            means = [np.mean(x, axis=0) for x in test_embeddings]
            std = [np.std(x, axis=0) for x in test_embeddings]
            for i in range(test_embeddings.shape[0]):
                test_embeddings[i] = (test_embeddings[i] - means[i]) / std[i]
        if config['y_strategy'] == 'scale_y':
            scaler = StandardScaler()
            train_labels = scaler.fit_transform(train_labels.reshape(-1, 1))
            test_labels = scaler.transform(test_labels.reshape(-1, 1))
        import resource
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        folder = config['folder_fn'] + config['embedding'] + '_'
        print("model", model_name)
        resnet_mode = None
        if model_name in ['dense_net', 'resnet50', 'resnet101']:
            train_embeddings = np.array([np.pad(x,
                                                pad_width=((0, 0), (1, 1)),
                                                mode='constant', constant_values=0) for x in train_embeddings])
            test_embeddings = np.array([np.pad(x,
                                               pad_width=((0, 0), (1, 1)),
                                               mode='constant', constant_values=0) for x in test_embeddings])
            

        model = initialize_model(config, model_name, train_embeddings)
        monitor_loss = 'val_mae'
        if config['n_categories'] > 0 and config['multi_output'] == None:
            monitor_loss = 'val_loss'
        elif config['multi_output']:
            monitor_loss = 'val_regression_mae'
        model_callbacks = [
            GCCallback(),
            WandbCallback(),
            EarlyStopping(
                monitor=monitor_loss,
                patience=config['patience'],
                verbose=0,
                mode='auto',
                baseline=None,
                restore_best_weights=True)
        ]
        if config['lr_plateau']:
            reduce_lr = ReduceLROnPlateau(monitor=monitor_loss,
                                          factor=config['lr_plateau_factor'],
                                          patience=config['lr_plateau_patience'],
                                          min_lr=config['min_lr'],
                                          cooldown=config['cooldown']
                                         )
            model_callbacks.append(reduce_lr)
        weights = None
        if config['multi_output']:
            test_labels, train_labels, label_encoder, weights = multioutput_y_preparation(test_labels, train_labels, config['n_categories'], config['class_weights'])


        for i, x in enumerate(test_embeddings):
            start_point = np.random.randint(0, x.shape[0] - config['mfcc_shape'][0] + 1)
            end_point = start_point + config['mfcc_shape'][0]
            x = x[start_point:end_point, :]
            if preprocessing_strategy == 'norm_mfcc_dataloader':
                x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
            elif preprocessing_strategy == 'sub_mean_dataloader':
                x = x - np.mean(x, axis=0)
            test_embeddings[i] = x
        test_embeddings = np.stack(test_embeddings)
        if config['model_name'] not in ['lstm', 'vanilla_lstm', 'cnn_custom_1d', 'lstm_cnn', 'tdnn_model',
                                        'cnn_resnet_1d']:
            test_embeddings = test_embeddings.reshape(
                test_embeddings.shape[0],
                test_embeddings.shape[1],
                test_embeddings.shape[2],
                1
            )
            if config['model_name'] in ['resnet50', 'resnet101']:
                resnet_mode = True
                test_embeddings = np.repeat(test_embeddings, 3, axis=3)
        print("Train", train_embeddings.shape, "Test", test_embeddings.shape, "Resnet mode", resnet_mode)
        train_gen = DataGenerator(train_embeddings,
                                  train_labels,
                                  config['mfcc_shape'][0],
                                  preprocessing_strategy=preprocessing_strategy,
                                  preprocess_values=preprocess_values,
                                  batch_size=config['batch_size'],
                                  resnet_mode=resnet_mode,
                                  shuffle_temporal=config['shuffle_temporal'],
                                  model_name=model_name,
                                  multi_output=config['multi_output'])
        for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                         key= lambda x: -x[1])[:10]:
            print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
        
        if config['model_name'] == 'lstm_cnn':
            train_lstm_cnn(config, model, model_callbacks, test_embeddings, test_labels, train_gen)

        else:
            """
            print("Before model.fit")
            time.sleep(10)"""
            model.fit(train_gen,
                      validation_data=(test_embeddings, test_labels),
                      epochs=config['epochs'],
                      verbose=1,
                      callbacks=model_callbacks,
                      class_weight=weights
                      )
        y_pred = model.predict(test_embeddings)
        
        if config['multi_output']:
            y_pred_label = np.argmax(y_pred[1], axis=-1)
            y_pred = y_pred[0]
            y_pred_label = label_encoder.inverse_transform(y_pred_label)
            print(y_pred_label)
            np.savetxt(folder + model_name + config['timestamp'] + '-y_pred-labels' + str(fold_iteration) + '.txt',
                   y_pred_label,
                   fmt="%s")
            test_labels = test_labels[0]
        elif config['n_categories'] > 0:
            y_pred = np.argmax(y_pred, axis=-1)
            test_labels = np.argmax(test_labels, axis=-1)
            
        if config['y_strategy'] == 'scale_y':
            test_labels = scaler.inverse_transform(test_labels)
            y_pred = scaler.inverse_transform(y_pred)

        fold_mae = sklearn.metrics.mean_absolute_error(y_pred=y_pred, y_true=test_labels)
        print('Fold', fold_iteration, 'MAE', fold_mae)
        mae_per_pred = np.abs(y_pred - test_labels)
        mae_per_obs = np.std(mae_per_pred)
        wandb.log(
            {
                'mae': fold_mae,
                'fold': fold_iteration,
                'fold_std_mae': mae_per_obs
            }
        )
        
        
        maes.append(fold_mae)
        std_maes.append(mae_per_obs)

        tf.keras.backend.clear_session()
        del model
        np.savetxt(folder + model_name + config['timestamp'] + '-' + 'y_true' + str(fold_iteration) + '.txt',
                   np.array(test_labels))
        np.savetxt(folder + model_name + config['timestamp'] + '-y_pred' + str(fold_iteration) + '.txt',
                   np.array(y_pred))
        np.savetxt(folder + model_name + config['timestamp'] + '-spk_labels' + str(fold_iteration) + '.txt',
                   X_spk_labels[test_index],
                   fmt="%s")

        fold_iteration += 1

    wandb.log(
        {
            'Final MAE': np.mean(maes, dtype=np.float64),
            'MAE std': np.std(maes, dtype=np.float64),
            'Final_fold_std - average': np.mean(std_maes)
        }
    )
    print('Final MAE', np.mean(maes), 'MAE std', np.std(maes))


def multioutput_y_preparation(test_labels, train_labels, n_categories, class_weights):
    print("multioutput_y_preparation >>>")
    import pandas as pd
    if n_categories == 8:
        bins = [0, 20, 30, 35, 40, 45, 60, 80, np.inf]
        names = ['<20', '20-30', '30-35', '35-40', '40-45', '45-60', '60-80', '80+']
    elif n_categories == 9:
        bins = [0, 25, 30, 35, 40, 45, 50, 60, 70, np.inf]
        names = ['<25', '25-30', '30-35', '35-40', '40-45','45-50', '50-60', '60-70', '70+']
    train_labels_bins = pd.cut(train_labels, bins, labels=names)
    test_labels_bins = pd.cut(test_labels, bins, labels=names)
    
    
    # Transform categories in appropriate one-hot vectors
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoder.fit(np.array(names).reshape(-1, 1))
    train_labels_bins = encoder.transform(np.array(train_labels_bins).reshape(-1, 1))
    print("Train label encoder output:", train_labels_bins.shape)
    test_labels_bins = encoder.transform(np.array(test_labels_bins).reshape(-1, 1))
    print("Test label encoder output:", test_labels_bins.shape)
    from tensorflow.keras.utils import to_categorical
    train_labels_bins = to_categorical(train_labels_bins, num_classes=len(names))
    test_labels_bins = to_categorical(test_labels_bins, num_classes=len(names))
    weights = None
    train_labels = [train_labels, train_labels_bins]
    test_labels = [test_labels, test_labels_bins]
    if class_weights:
        y_integers = np.argmax(train_labels_bins, axis=-1)
        from sklearn.utils import class_weight
        weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
        # Concatenate the two labels of interest
        
        weights = dict(enumerate(weights))
        weights = {'classifier': weights}
        print(weights)
        print("multioutput_y_preparation <<<")
    return test_labels, train_labels, encoder, weights


def train_lstm_cnn(config, model, model_callbacks, test_embeddings, test_labels, train_gen):
    model.fit(train_gen,
              validation_data=(test_embeddings, test_labels),
              epochs=config['epochs_all'],
              verbose=1,
              callbacks=model_callbacks
              )
    # Freeze lstm layers
    for layer in model.layers[0:2]:
        print(layer)
        layer.trainable = False
    # Change LR
    from tensorflow.keras import backend as K
    K.set_value(model.optimizer.lr, config['lr'] / 10)
    print("New LR:", K.get_value(model.optimizer.lr))
    # Train model
    model.fit(train_gen,
              validation_data=(test_embeddings, test_labels),
              epochs=config['epochs'],
              verbose=1,
              callbacks=model_callbacks
              )


def initialize_model(config, model_name, train_embeddings):
    print("Initialize model >>>")
    if model_name == 'cnn_four_blocks':
        model = cnn_four_blocks(
            input_shape=(config['mfcc_shape'][0], train_embeddings[0].shape[1], 1),
            apply_dropout=config['dropout'],
            config=config
        )
    elif model_name == 'cnn_three_blocks':
        model = cnn_three_blocks(
            input_shape=(config['mfcc_shape'][0], train_embeddings[0].shape[1], 1),
            apply_dropout=config['dropout'],
            config=config
        )
    elif model_name == 'cnn_custom':
        model = cnn_custom(
            input_shape=(config['mfcc_shape'][0], train_embeddings[0].shape[1], 1),
            apply_dropout=config['dropout'],
            config=config
        )
    elif model_name == 'cnn_resnet':
        model = create_res_net(
            input_shape=(config['mfcc_shape'][0], train_embeddings[0].shape[1], 1),
            config=config
        )
    elif model_name == 'cnn_resnet_1d':
        model = create_res_net_1d(
            input_shape=(config['mfcc_shape'][0], train_embeddings[0].shape[1]),
            config=config
        )
    elif model_name == 'cnn_custom_1d':
        model = cnn_custom_1d(
            input_shape=(config['mfcc_shape'][0], train_embeddings[0].shape[1]),
            apply_dropout=config['dropout'],
            config=config
        )
    elif model_name == 'lasso_custom':
        model = lasso_custom(
            input_shape=(config['mfcc_shape'][0], train_embeddings[0].shape[1], 1),
            config=config
        )
    elif model_name == 'fc_custom':
        model = fc_custom(
            input_shape=(config['mfcc_shape'][0], train_embeddings[0].shape[1], 1),
            config=config
        )    
    elif model_name == 'tdnn_model':
        model = tdnn_model(
            input_shape=(config['mfcc_shape'][0], train_embeddings[0].shape[1], 1),
            config=config
        )
    elif model_name == 'cnn_two_conv_one_max':
        model = cnn_two_conv_one_max(
            input_shape=(config['mfcc_shape'][0], train_embeddings[0].shape[1], 1),
            apply_dropout=config['dropout'],
            config=config
        )

    elif model_name == 'lstm':
        model = lstm_model(
            input_shape=(config['mfcc_shape'][0], train_embeddings[0].shape[1]),
            config=config
        )
    elif model_name == 'vanilla_lstm':
        model = vanilla_lstm_model(
            input_shape=(config['mfcc_shape'][0], train_embeddings[0].shape[1]),
            config=config
        )
    elif model_name == 'lstm_cnn':
        model = lstm_cnn(
            input_shape=(config['mfcc_shape'][0], train_embeddings[0].shape[1]),
            config=config
        )
    elif model_name in ['dense_net', 'resnet50', 'resnet101']:
        if model_name == 'dense_net':
            channels = 1
        else:
            channels = 3
        
        model = pretrained_model(
            input_shape=(config['mfcc_shape'][0], train_embeddings[0].shape[1], channels),
            config=config)
    print("Initialize model <<<")
    return model


def oversample_dataset(train_embeddings, train_labels):
    print("oversample_dataset >>>")
    idx = {}
    age_freq = {}
    # find most frequent age
    for age in np.unique(train_labels):
        current_idx = np.where(train_labels == age)[0]
        age_freq[age] = current_idx.shape[0]
        idx[age] = current_idx
    max_freq = np.max(list(age_freq.values()))
    # Iterate once again on ages in order to perform oversample
    new_idx = []
    for age in np.unique(train_labels):
        if age_freq[age] < max_freq:
            oversampled_idx = np.random.randint(low=0, high=age_freq[age], size=max_freq)
            current_idx = idx[age][oversampled_idx]
        else:
            current_idx = idx[age]
        new_idx.append(current_idx)
    new_idx = np.concatenate(new_idx)
    print("Before oversampling:", train_embeddings.shape)
    train_embeddings = train_embeddings[new_idx,]
    train_labels = train_labels[new_idx,]
    print("After oversampling:", train_embeddings.shape)
    print("oversample_dataset <<<")
    return train_embeddings, train_labels


def undersample_dataset(config, train_embeddings, train_labels):
    print("undersample_dataset >>>")
    idx = {}
    max_freq = config['max_freq']
    for age in np.unique(train_labels):
        current_idx = np.where(train_labels == age)[0]
        age_freq = current_idx.shape[0]
        if age_freq > max_freq:
            undersampled_idx = np.random.randint(low=0, high=age_freq, size=max_freq)
            idx[age] = current_idx[undersampled_idx]
        else:
            idx[age] = current_idx
    new_idx = np.concatenate(list(idx.values()))
    print("Before undersampling:", train_embeddings.shape)
    train_embeddings = train_embeddings[new_idx, ]
    train_labels = train_labels[new_idx, ]
    print("After undersampling:", train_embeddings.shape)
    print("undersample_dataset <<<")
    return train_embeddings, train_labels


def train_holdout(X_train,
                  y_train,
                  X_test,
                  y_test,
                  X_aug,
                  y_aug,
                  preprocessing_strategy,
                  model_name,
                  config):


    if config['unbalanced'] and config['data_augmentation']:
        print("Shape before:", X_train.shape)
        X_train = np.concatenate((X_train, X_aug))
        print("Shape after:", X_train.shape)
        y_train = np.concatenate((y_train, y_aug))

    if config['include_title_only_obs']:
        if config['unbalanced_include_title_only_obs']:
            print("Loading unbalanced_include_title_only_obs")
            data = np.load('/media/hdd1/khaled/npz_files/numpy_train_test_obj_unbalanced-title_only.npz', allow_pickle=True)
        else:
            data = np.load('/media/hdd1/khaled/npz_files/numpy_train_test_obj-title_only.npz', allow_pickle=True)

        vectors = []
        for x in list(data.keys()):
            vectors.append(data[x])

        X_title_only, y_title_only, X_spk_labels_title_only, X_spk_labels_aug_title_only, X_aug_title_only, y_aug_title_only = vectors
        if config['embedding'] == 'mel_spect_kaldi':
            X_title_only_mel = empty(X_title_only.shape, dtype='object')
            for i in range(X_title_only.shape[0]):
                temp = scipy.fftpack.idct(X_title_only[i])
                X_title_only_mel[i] = temp
            X_title_only = X_title_only_mel
            X_aug_title_only_mel = empty(X_aug_title_only.shape, dtype='object')
            for i in range(X_aug_title_only.shape[0]):
                X_aug_title_only_mel[i] = scipy.fftpack.idct(X_aug_title_only[i])
            X_aug_title_only = X_aug_title_only_mel
        print("Shape before include_title_only_obs:", X_train.shape)
        X_train = np.concatenate((X_train, X_title_only))
        X_train = np.concatenate((X_train, X_aug_title_only))
        print("Shape after include_title_only_obs:", X_train.shape)
        y_train = np.concatenate((y_train, y_title_only))
        y_train = np.concatenate((y_train, y_aug_title_only))
        data.close()
        data = None
        del data
        del X_title_only, y_title_only, X_spk_labels_title_only, X_spk_labels_aug_title_only, X_aug_title_only, y_aug_title_only, vectors
        import gc
        gc.collect()

    scaler = None
    preprocess_values = None
    print("Preprocess strat:", preprocessing_strategy)
    if preprocessing_strategy == 'norm_mfcc':
        mean = np.array([np.mean(x, axis=0) for x in X_train])
        mean = np.mean(mean, axis=0)
        mean = mean.reshape(1, 1, mean.shape[0], 1)

        std = np.array([np.std(x, axis=0) for x in X_train])
        std = np.mean(std, axis=0)
        std = std.reshape(1, 1, std.shape[0], 1)
        preprocess_values = (mean, std)
    elif preprocessing_strategy == 'min_max':
        max_x = np.array([np.max(x, axis=0) for x in X_train])
        max_x = np.max(max_x, axis=0)
        max_x = max_x.reshape(1, 1, max_x.shape[0], 1)

        min_x = np.array([np.min(x, axis=0) for x in X_train])
        min_x = np.min(min_x, axis=0)
        min_x = min_x.reshape(1, 1, min_x.shape[0], 1)
        preprocess_values = (max_x, min_x)
    elif preprocessing_strategy == 'sub_mean':
        means = [np.mean(x, axis=0) for x in X_train]
        for i in range(X_train.shape[0]):
            X_train[i] = X_train[i] - means[i]
        means = [np.mean(x, axis=0) for x in X_test]
        for i in range(X_test.shape[0]):
            X_test[i] = X_test[i] - means[i]
    elif preprocessing_strategy == 'sub_mean_div_std':
        means = [np.mean(x, axis=0) for x in X_train]
        std = [np.std(x, axis=0) for x in X_train]
        for i in range(X_train.shape[0]):
            X_train[i] = (X_train[i] - means[i]) / std[i]
        means = [np.mean(x, axis=0) for x in X_test]
        std = [np.std(x, axis=0) for x in X_test]
        for i in range(X_test.shape[0]):
            X_test[i] = (X_test[i] - means[i]) / std[i]
    if config['y_strategy'] == 'scale_y':
        scaler = StandardScaler()
        y_train = scaler.fit_transform(y_train.reshape(-1, 1))
        y_test = scaler.transform(y_test.reshape(-1, 1))
    import resource
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    folder = config['folder_fn'] + config['embedding'] + '_'
    print("model", model_name)
    resnet_mode = None
    if model_name in ['dense_net', 'resnet50', 'resnet101']:
        X_train = np.array([np.pad(x,
                                            pad_width=((0, 0), (1, 1)),
                                            mode='constant', constant_values=0) for x in X_train])
        X_test = np.array([np.pad(x,
                                           pad_width=((0, 0), (1, 1)),
                                           mode='constant', constant_values=0) for x in X_test])

    model = initialize_model(config, model_name, X_train)
    monitor_loss = 'val_mae'
    if config['n_categories'] > 0 and config['multi_output'] == None:
        monitor_loss = 'val_loss'
    elif config['multi_output']:
        monitor_loss = 'val_regression_mae'
    model_callbacks = [
        GCCallback(),
        WandbCallback(),
        EarlyStopping(
            monitor=monitor_loss,
            patience=config['patience'],
            verbose=0,
            mode='auto',
            baseline=None,
            restore_best_weights=True)
    ]
    if config['lr_plateau']:
        reduce_lr = ReduceLROnPlateau(monitor=monitor_loss,
                                      factor=config['lr_plateau_factor'],
                                      patience=config['lr_plateau_patience'],
                                      min_lr=config['min_lr'],
                                      cooldown=config['cooldown']
                                      )
        model_callbacks.append(reduce_lr)
    weights = None
    if config['multi_output']:
        y_test, y_train, label_encoder, weights = multioutput_y_preparation(y_test, y_train,
                                                                                      config['n_categories'],
                                                                                      config['class_weights'])

    for i, x in enumerate(X_test):
        start_point = np.random.randint(0, x.shape[0] - config['mfcc_shape'][0] + 1)
        end_point = start_point + config['mfcc_shape'][0]
        x = x[start_point:end_point, :]
        if preprocessing_strategy == 'norm_mfcc_dataloader':
            x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
        elif preprocessing_strategy == 'sub_mean_dataloader':
            x = x - np.mean(x, axis=0)
        X_test[i] = x
    X_test = np.stack(X_test)
    if config['model_name'] not in ['lstm', 'vanilla_lstm', 'cnn_custom_1d', 'lstm_cnn', 'tdnn_model',
                                    'cnn_resnet_1d']:
        X_test = X_test.reshape(
            X_test.shape[0],
            X_test.shape[1],
            X_test.shape[2],
            1
        )
        if config['model_name'] in ['resnet50', 'resnet101']:
            resnet_mode = True
            X_test = np.repeat(X_test, 3, axis=3)
    print("Train", X_train.shape, "Test", X_test.shape, "Resnet mode", resnet_mode)
    train_gen = DataGenerator(X_train,
                              y_train,
                              config['mfcc_shape'][0],
                              preprocessing_strategy=preprocessing_strategy,
                              preprocess_values=preprocess_values,
                              batch_size=config['batch_size'],
                              resnet_mode=resnet_mode,
                              shuffle_temporal=config['shuffle_temporal'],
                              model_name=model_name,
                              multi_output=config['multi_output'])

    model.fit(train_gen,
              validation_data=(X_test, y_test),
              epochs=config['epochs'],
              verbose=1,
              callbacks=model_callbacks,
              class_weight=weights
              )

    return model