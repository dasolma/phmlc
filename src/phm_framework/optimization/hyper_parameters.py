import inspect
from hashlib import sha1

import phm_framework as phmf
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
import tensorflow as tf

import os
import logging

from phm_framework import scoring, typing
from phm_framework.utils import flat_dict

logging.basicConfig(level=logging.INFO)
logging.info("Working dir: " + os.getcwd())

OUTPUT = [
    {
        'field': 'target',
        'value': 'rul',
        'output': 'relu'
    },
    {
        'field': 'type',
        'value': 'classification:binary',
        'output': 'sigmoid'
    },
    {
        'field': 'type',
        'value': 'classification:multiclass',
        'output': lambda task: 'sigmoid' if isinstance(task['target'], list) else 'softmax'
    },

]

LOSS_METRICS = [
    {
        'field': 'type',
        'value': 'regression',
        'output': [tf.keras.losses.MeanSquaredError(name='mse'),
                   lambda task: tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                   lambda task: tf.keras.metrics.MeanAbsolutePercentageError(name="mape"),
                   lambda task: tf.keras.metrics.MeanAbsoluteError(name="mae"),
                   lambda task: scoring.SMAPE(name="smape"),
                   lambda task: scoring.NASAScore(name="nasa_score"),
                   ]
    },
    {
        'field': 'type',
        'value': 'classification:binary',
        'output': [tf.keras.losses.BinaryCrossentropy(name='cross_entropy'),
                   lambda task: tf.keras.metrics.BinaryAccuracy(name='acc'),
                   lambda task: phmf.models.utils.Recall(len(task['target_labels']), mode='macro', name='recall'),
                   lambda task: phmf.models.utils.Precision(len(task['target_labels']), mode='macro', name='precision')
                   ]
    },
    {
        'field': 'type',
        'value': 'classification:multiclass',
        'output': [lambda task: (tf.keras.losses.CategoricalCrossentropy(name='cross_entropy')
                                 if isinstance(task['target'], list)
                                 else tf.keras.losses.SparseCategoricalCrossentropy(name='cross_entropy')),
                   lambda task: (None
                                 if isinstance(task['target'], list)
                                 else tf.keras.metrics.SparseCategoricalAccuracy(name='acc')),
                   lambda task: phmf.models.utils.NonExclusiveRecall(len(task['target_labels']),
                                                                mode='macro', name='recall')
                                    if isinstance(task['target'], list)
                                    else phmf.models.utils.Recall(len(task['target_labels']),
                                                                mode='macro', name='recall'),
                   lambda task: phmf.models.utils.NonExclusivePrecision(len(task['target_labels']),
                                                                mode='macro', name='precision')
                                    if isinstance(task['target'], list)
                                    else phmf.models.utils.Precision(len(task['target_labels']),
                                                                   mode='macro', name='precision')
         ]
    },
]



OUTPUT_DIM = [
    {
        'field': 'type',
        'value': 'regression',
        'output': 1
    },
    {
        'field': 'type',
        'value': 'classification:binary',
        'output': 1
    },
    {
        'field': 'type',
        'value': 'classification:multiclass',
        'output': lambda task: len(task['target_labels'])
    },

]


RANGES = {
    # nlp
    'nhideen_layers': (1, 7),
    'activation': (-0.49, len(phmf.typing.ACTIVATIONS) + 0.49 - 1),

    # mscnn
    'kernel_size': (lambda task: min(3, len(task['features'])) + .3,
                    lambda task: min(10, len(task['features'])) + 0.99),
    'msblocks': (-0.49, 3.49),
    'block_size': (-0.49, 5.49),
    'f1': (lambda task: min(3, task['min_ts_len']),
           lambda task: min(20, task['min_ts_len'])),
    'f2': (lambda task: min(3, task['min_ts_len']),
           lambda task: min(20, task['min_ts_len'])),
    'f3': (lambda task: min(3, task['min_ts_len']),
           lambda task: min(20, task['min_ts_len'])),
    'filters': (16, 64),
    'conv_activation': (-0.49, len(phmf.typing.ACTIVATIONS) + 0.49 - 1),

    'dilation_rate': (-0.49, 10.49),


    # rnn
    'cell_type': (-0.49, 1.49),
    'rnn_units': (32, 256),
    'bidirectional': (0, 1),

    # general
    'nblocks': (0.51, 4.49),
    'fc1': (16, 64),
    'fc2': (16, 64),
    'dense_activation': (-0.49, len(phmf.typing.ACTIVATIONS) + 0.49 - 1),
    'batch_normalization': (0, 1),

    # regularization
    'dropout': (0, 0.1),
    'l1': (0, 0.00001),
    'l2': (0, 0.00001),

    # transformers
    'nlayers': (0.51, 3.49),
    'segment_size': (0.05, 0.25),
    'model_dim': (8, 64),
    'num_heads': (8, 64),
    'mlp_dim': (16, 64),

    # random forest
    'n_estimators': (1.0, 1000),
    'min_samples_split': (0.01, 0.1),
    'max_features': (0.1, 1.0),
    'max_samples': (0.1, 1.0),
    'min_child_weight': (1.0, 3.0),

    # svm
    'tol': (1e-5, 1e-3),
    'C':  (0, 1.0),


}

class DummyPreprocess():

    def fit(self, X):
        pass

    def transform(self, X):
        return X

PREPROCESS = {
    None: DummyPreprocess,
    'norm': MinMaxScaler,
    'std': StandardScaler,
}

POINTS_TO_EVALUATE = {
    'fcn': [{'data': {'low_float_precision': True},
             'model__batch_normalization': 1,
             'model__nhideen_layers': 2,
             'model__activation': 0,
             'model__dropout': 0.05,
             'model__l1': 1e-4,
             'model__l2': 1e-4,
             'train__lr': 0.0001,
             'train__stride': 0.2,
             'train__ts_len': lambda task: min(512, task['min_ts_len'] // 2)}],
    'rnn': [{'data': {'low_float_precision': True},
             'model__batch_normalization': 1,
             'model__bidirectional': 0,
             'model__cell_type': 0,
             'model__dense_activation': 0,
             'model__dropout': 0.05,
             'model__fc1': 128,
             'model__fc2': 64,
             'model__l1': 1e-4,
             'model__l2': 1e-4,
             'model__nblocks': 3,
             'model__rnn_units': 128,
             'train__lr': 0.0001,
             'train__stride': 0.2,
             'train__ts_len': lambda task: min(512, task['min_ts_len'] // 2)}],
    'mscnn': [{'data': {'low_float_precision': True},
               'model__batch_normalization': 1,
               'model__block_size': 2,
               'model__nblocks': 2,
               'model__msblocks': 2,
               'model__kernel_size': 1.10,
               'model__filters': 64,
               'model__f1': 10,
               'model__f2': 15,
               'model__f3': 20,
               'model__dense_activation': 0,
               'model__conv_activation': 0,
               'model__dropout': 0.05,
               'model__fc1': 256,
               'model__fc2': 128,
               'model__l1': 1e-4,
               'model__l2': 1e-4,
               'train__lr': 0.0001,
               'train__stride': 0.2,
               'train__ts_len': lambda task: min(512, task['min_ts_len'] // 2)}],
}

DATASET_FIXED_PARAMS = {
    'CWRU': {
        'train': {
            'stride': 0,
        },

        'data': {
            'preprocess': None,
        },

        'model': {
            'l1': 0,
            'l2': 0,
            'dropout': 0,
        }
    },

    'DFD15': {
        'train': {
            'stride': 0,
        },

        'data': {
            'preprocess': 'std',
        },

        'model': {
            'l1': 0,
            'l2': 0,
            'dropout': 0,
        }
    },

    'JNUB': {
        'train': {

        },

        'data': {
            'preprocess': 'std',
        },

        'model': {

        }
    }
}






def remove_fixedparam(params, dataset_name):

    if dataset_name in DATASET_FIXED_PARAMS:
        fixed_params = flat_dict(DATASET_FIXED_PARAMS[dataset_name].copy())

        for key in fixed_params.keys():

            if key in params:
                del params[key]

    return params

def update_dict(d1, d2, task):
    keys = list(d2.keys())

    if len(keys) == 0:
        return d1

    key = keys[0]
    value = d2[key]
    if key in d1:

        if isinstance(value, dict):
            d1[key] = update_dict(d1[key], value, task)

        elif callable(value):
            d1[key] = value(task)
        else:
            d1[key] = value

    else:
        if callable(value):
            d1[key] = value(task)
        else:
            d1[key] = value

    d2 = d2.copy()
    del d2[key]

    value = d1[key]
    del d1[key]

    r = update_dict(d1, d2, task)
    r[key] = value

    return r


def get_points_to_evaluate(optimization_config, task):
    net = optimization_config['model']['net']
    dataset_name = optimization_config['data']['dataset_name']

    if net in POINTS_TO_EVALUATE.keys():
        result = [update_dict(optimization_config, remove_fixedparam(point, dataset_name), task)
                  for point in POINTS_TO_EVALUATE[net]]

        logging.info(f"Settting points to evaluate: {result}")

        return result

    return []


def get_config(task, rules):
    for rule in rules:
        if task[rule['field']] == rule['value']:
            o = rule['output']

            o = compute_value(o, task)

            return o

def is_task_required(o):
    return callable(o) and 'task' in list(inspect.signature(o).parameters.keys())

def compute_value(o, task):
    if isinstance(o, list):
        o = [e(task) if is_task_required(e) else e for e in o]
    elif is_task_required(o):
        o = o(task)
    return o


def get_loss(task):
    return get_config(task, LOSS_METRICS)


def get_output(task):
    return get_config(task, OUTPUT)


def get_output_dim(task):
    return get_config(task, OUTPUT_DIM)


def compute_ranges(task):
    return {k: (compute_value(v1, task), compute_value(v2, task)) for k, (v1, v2) in RANGES.items()}
