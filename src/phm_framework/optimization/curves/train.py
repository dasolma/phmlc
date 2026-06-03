import importlib
import logging
import sys
import time
import traceback
import numpy as np
import tqdm
import random
import os
from matplotlib import pyplot as plt
from phmd import datasets
from phm_framework.data import Sequence, SequenceV2, FSLSequence
from phm_framework.data.generators import load_train_generators, load_train_net_generators_v2
from phm_framework.logging import log_train, HASH_EXCLUDE, confighash, secure_decode

from phm_framework.optimization.hyper_parameters import flat_dict
from phm_framework.optimization.utils import load_log, train_rule_tree
from statsmodels.tsa.arima.model import ARIMA
import phm_framework as phmf
import warnings
from sklearn import model_selection, tree
import pandas as pd
from sklearn.tree import _tree, export_text
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from phm_framework.trainers.utils import get_task
import pickle as pk
import copy

warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)


class CurvesConditionedSequence(Sequence):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._cond = None
        self.__units = self.units

    @property
    def cond(self):
        return self._cond

    @cond.setter
    def cond(self, value):
        self._cond = value
        self.units = list(set(self._cond.keys()).intersection(self.__units))

    def init_batch_matrices(self, ts_len):
        k = list(self._cond.keys())[0]
        features_shape = self._cond[k].shape[::-1]
        if self.extra_channel:
            X = [np.zeros(shape=(self.batch_size, ts_len, self.nfeatures, 1)),
                 np.zeros(shape=(self.batch_size,) + features_shape + (1,))]
        else:
            X = [np.zeros(shape=(self.batch_size, ts_len, self.nfeatures)),
                 np.zeros(shape=(self.batch_size,) + features_shape)]

        # initialize target matrix
        if isinstance(self.target_col, list):
            Y = np.zeros(shape=(self.batch_size, len(self.target_col)))
        else:
            Y = np.zeros(shape=(self.batch_size,))
        return X, Y

    def update_batch_matrices(self, Db, T, X, Y, i, indexes, k, ts_len, unit):
        if self.extra_channel:
            X[0][i, :, :, 0] = Db[indexes, :]
            X[1][i, :, :, 0] = self.cond[unit].T
        else:
            X[0][i, :, :] = Db[indexes, :]
            X[1][i, :, :] = self.cond[unit].T
        # select last point of sequence target
        Y[i] = T[k + ts_len]


def curves_train(model_creator, config, ifold, queue, debug, directory, timeout):
    logging.info('Starting training (fold %d) %s' % (ifold, config))

    try:
        training_config = config['train']
        net_config = config['model']
        data_config = config['data']

        net_name = net_config['net']
        data_name = data_config['dataset_name']
        data_target = data_config['dataset_target']

        task = get_task(data_name, data_target, model_creator)

        csv_config = flat_dict(config.copy())
        csv_config['train__max_epochs'] = csv_config.pop('train__epochs')
        csv_config['train__fold'] = ifold
        nhash = confighash(csv_config, exclude=HASH_EXCLUDE)
        arch_hash = confighash(csv_config, exclude=HASH_EXCLUDE + ["train__fold"])
        csv_config['run_hash'] = nhash
        csv_config['arch_hash'] = arch_hash

        import os
        import tensorflow as tf
        from phm_framework import models
        from phm_framework.models.utils import AdditionalRULValidationSets
        from phm_framework.optimization import hyper_parameters as hp

        # prepare output directory
        if not os.path.exists(directory):
            os.makedirs(directory)

        log_csv = load_log(None, directory)
        if not isinstance(log_csv, bool):
            query = log_csv[log_csv.run_hash == nhash]
            if query.shape[0] > 0 and query.iloc[0].train__status == 'FINISHED':
                r = query.iloc[0]
                queue.put(({
                               'val_loss': [r.val_loss],
                               'test_loss': [r.test_loss]
                           }, arch_hash))
                return

        # data reading and prepare data generators
        logging.info("Reading data")
        ts_len = secure_decode(training_config, "ts_len", dtype=int, task=task)
        random_state = secure_decode(training_config, "random_state", dtype=int, task=task)
        conditioning = secure_decode(training_config, "conditioning", dtype=str, default=None, task=task)
        preprocess = hp.PREPROCESS[secure_decode(data_config, "preprocess", str, default='norm', task=task)]()

        if config['model']['net'] == 'protonet':
            sequencer = FSLSequence
        else:
            sequencer = Sequence
        sets = load_train_generators(data_name,
                                     task_name=data_target,
                                     ts_len=ts_len, fold=ifold, num_folds=config['train']['num_folds'],
                                     preprocess=preprocess, return_test=True,
                                     normalize_output=False,
                                     filters={"data": "curves"},
                                     random_state=random_state,
                                     return_train_val=True,
                                     sequencer=sequencer)

        train_gen = sets['train']
        val_gen = sets['val']
        test_gen = sets['test']

        ds = datasets.Dataset(data_name)
        _task = ds['final_loss']
        _task.filters = {"data": "results"}

        (results,) = _task.load()

        nfeatures = 0
        if conditioning == 'net':
            activations = [a if isinstance(a, str) else a.__class__.__name__ for a in phmf.typing.ACTIVATIONS]
            rnn_cells = [a.__name__ for a in phmf.typing.RNN_CELLS]

            def get_code(x, elements):
                x = str(x)
                for i, a in enumerate(elements):
                    if a in x:
                        return i

                return float(x)

            exclude = ['model__input_shape', 'model__output', 'model__net']
            features = [c for c in results.columns if 'model__' in c if c not in exclude]

            results['model__activation'] = results['model__activation'].map(
                lambda x: x if x is np.nan else round(get_code(x, activations)))
            results['model__dense_activation'] = results['model__dense_activation'].map(
                lambda x: x if x is np.nan else round(get_code(x, activations)))
            results['model__conv_activation'] = results['model__conv_activation'].map(
                lambda x: x if x is np.nan else round(get_code(x, activations)))
            results['model__cell_type'] = results['model__cell_type'].map(
                lambda x: x if x is np.nan else round(get_code(x, rnn_cells)))
            results['model__kernel_size'] = results['model__kernel_size'].map(
                lambda x: (x if x is np.nan else float(x)) if '(' not in str(x) else eval(x)[0] + eval(x)[1] / 100)
            results['model__batch_normalization'] = results['model__batch_normalization'].map(
                lambda x: x if x is np.nan else round(float(eval(str(x)))))
            results['model__bidirectional'] = results['model__bidirectional'].map(
                lambda x: x if x is np.nan else round(float(eval(str(x)))))

            for feature in features:
                f = results[feature]
                results[feature] = (f - f.min()) / (f.max() - f.min())

            net_info = results[features + ['unit']].groupby('unit').first().to_dict('index')
            net_info = {k: np.array([f[kf] for kf in features]) for k, f in net_info.items()}
            net_info = {k: np.array([np.nan_to_num(f), np.isnan(f)]) for k, f in net_info.items()}
            train_gen.cond = net_info
            val_gen.cond = net_info
            test_gen.cond = net_info

            nfeatures = len(features)

        stride = 1
        train_gen.stride = stride
        val_gen.stride = stride
        test_gen.stride = stride

        # batch size
        batch_size = secure_decode(training_config, "batch_size", dtype=int, task=task)
        train_gen.batch_size = batch_size
        val_gen.batch_size = 256
        test_gen.batch_size = 256

        extra_channel = getattr(importlib.import_module(model_creator.__module__), 'EXTRA_CHANNEL')
        train_gen.extra_channel = extra_channel
        val_gen.extra_channel = extra_channel
        test_gen.extra_channel = extra_channel

        if "train_generator" in config:
            for key, value in config["train_generator"].items():
                setattr(train_gen, key, value)
        if "val_generator" in config:
            for key, value in config["val_generator"].items():
                setattr(val_gen, key, value)

        if config['model']['net'] == 'protonet':
            input_shape = (100, 3)
        else:
            input_shape = (ts_len, 2)

        logging.info("Finished Data reading")

        # training config
        epochs = secure_decode(training_config, "epochs", int, task=task)
        epochs = min(5, epochs) if debug else epochs

        lr = secure_decode(training_config, "lr", float, pop=False, task=task)
        monitor = secure_decode(training_config, "monitor", str, default="val_loss", task=task)
        verbose = secure_decode(training_config, "verbose", bool, default=False, task=task)

        output_dim = hp.get_output_dim(task)
        output = hp.get_output(task)

        # create and compile model

        model_params = models.get_model_params(net_config, model_creator, task)
        csv_config.update(flat_dict({'model': model_params}))
        csv_config['model__output'] = output
        csv_config['model__output_dim'] = output_dim
        csv_config['model__input_shape'] = input_shape
        if 'num_features' in model_params:
            csv_config['model__num_features'] = nfeatures
            model_params['num_features'] = nfeatures
        del model_params['output_dim']
        del model_params['input_shape']
        del model_params['output']
        model = model_creator(input_shape, output_dim=output_dim, output=output,
                              **model_params)
        logging.info("Model created")
        model.summary(print_fn=lambda x: logging.info(x))

        model.compile(loss='mse',
                      metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                               tf.keras.metrics.MeanAbsoluteError(name="mae")],
                      optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      run_eagerly=False)

        # train
        es = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=8)
        rlr = tf.keras.callbacks.ReduceLROnPlateau(patience=3)

        extra_callbacks = []

        logging.info("Started training")
        start_time = time.time()
        train_keys = train_gen.data.keys()

        if config['model']['net'] == 'protonet':
            history = model.fit(train_gen, validation_data=val_gen,
                                batch_size=batch_size,
                                epochs=epochs, verbose=(2 if verbose else 0),
                                callbacks=[es, rlr] + extra_callbacks)
        else:
            if conditioning == 'net':
                train_keys = [k for k in train_keys if ''.join(k) in net_info.keys()]

            X_train = np.array([train_gen.data[k][:ts_len] for k in train_keys])
            steps_per_epoch = X_train.shape[0] // batch_size

            if conditioning == 'net':
                ninfo = np.array([net_info[''.join(k)].T for k in train_keys])
                X_train = (X_train, ninfo)
            Y_train = np.array([train_gen.target[k][0] for k in train_keys])

            val_keys = val_gen.data.keys()
            if conditioning == 'net':
                val_keys = [k for k in val_keys if ''.join(k) in net_info.keys()]
            X_val = np.array([val_gen.data[k][:ts_len] for k in val_keys])

            if conditioning == 'net':
                ninfo = np.array([net_info[''.join(k)].T for k in val_keys])
                X_val = (X_val, ninfo)
            Y_val = np.array([val_gen.target[k][0] for k in val_keys])
            history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                                batch_size=batch_size, steps_per_epoch=steps_per_epoch,
                                validation_steps=100,
                                epochs=epochs, verbose=(2 if verbose else 0),
                                callbacks=[es, rlr] + extra_callbacks)
        history = history.history

        # save csv opt_history
        csv_config['train__time'] = (time.time() - start_time)
        csv_config.update({k: history[k][-1] for k in history.keys() if k.startswith('val')})

        logging.info("Evaluating on test set")
        test_metrics = model.evaluate(test_gen, verbose=(2 if verbose else 0))
        for i, metric_name in enumerate(model.metrics_names):
            csv_config[f"test_{metric_name}"] = test_metrics[i]

        # simulate bayes optimization
        train_times = results[['unit', 'train__time']].set_index('unit').to_dict()['train__time']
        train_times = {tuple(k): train_times[k] for k in test_gen.units if k in train_times}

        assert len(train_times) > 0

        if nfeatures > 0:
            units = test_gen.units
            data = test_gen.data
            features = test_gen.cond
            epochs, signals, features = zip(
                *[(data[tuple(u)][:ts_len, :], data[tuple(u)], features[u].T) for u in units])
            inputs = [np.array(epochs), np.array(features)]
            units = [tuple(u) for u in units]
        else:
            units, inputs, signals = zip(*[(u, v[:ts_len, :], v) for u, v in test_gen.data.items()])
            inputs = np.array(inputs)

        preds = model.predict(inputs, verbose=1, batch_size=256)

        aux = {u: (s, p, train_times[u]) for u, s, p in zip(units, signals, preds) if u in train_times}
        test_datasets = set([u.split('_')[0] for u in test_gen.units])

        results = results[results.dataset.map(lambda x: x in test_datasets)]

        epochs_avoided = 0
        num_runs = 0
        num_prunings = 0
        total_epochs = 0
        total_train_time = 0
        avoided_train_time = 0
        experiments = results[['data__dataset_name', 'data__dataset_target', 'model__net']].drop_duplicates().values
        filter_best_losses = []
        real_best_losses = []
        for dataset, task, net in tqdm.tqdm(experiments):
            unit_ordered = list(
                results[(results.dataset == dataset) & (results.net == net) & (results.task == task)].unit)

            filter_best_loss = 100000
            real_best_loss = 10000

            for unit in unit_ordered:

                if unit in test_gen.units:
                    unit_data = aux[tuple(unit)]
                    real_val_loss = unit_data[0][:, 1][-1]
                    real_epochs = unit_data[0].shape[0]
                    pred_val_loss = unit_data[1][0]

                    train_time = unit_data[2]
                    total_train_time += train_time
                    total_epochs += real_epochs
                    num_runs += 1

                    if real_val_loss < real_best_loss:  # real best los found during real training
                        real_best_loss = real_val_loss

                    if pred_val_loss < filter_best_loss * 2:  # no stopped

                        if real_val_loss < filter_best_loss:  # best loss found during simulated training
                            filter_best_loss = real_val_loss

                    else:
                        epochs_avoided += (real_epochs - ts_len)
                        avoided_train_time += ((train_time / real_epochs) * (real_epochs - ts_len))
                        num_prunings += 1

            filter_best_losses.append(filter_best_loss)
            real_best_losses.append(real_best_loss)

        csv_config["epochs_avoided"] = epochs_avoided
        csv_config["total_epochs"] = total_epochs
        csv_config["total_train_time"] = total_train_time
        csv_config["avoided_train_time"] = avoided_train_time
        csv_config["filter_best_losses"] = filter_best_losses
        csv_config["real_best_losses"] = real_best_losses
        csv_config["num_prunings"] = num_prunings
        csv_config["num_runs"] = num_runs

        logging.info(f"epochs_avoided: {epochs_avoided}")
        logging.info(f"total_train_time: {total_train_time}")
        logging.info(f"avoided_train_time: {avoided_train_time}")
        logging.info(f"filter_best_losses: {filter_best_losses}")
        logging.info(f"real_best_losses: {real_best_losses}")
        logging.info(
            f"fount best: {np.mean([np.abs(f - r) < 1e-06 for f, r in zip(filter_best_losses, real_best_losses)])}")
        logging.info(f"num_prunings: {num_prunings}")
        logging.info(f"num_runs: {num_runs}")

        csv_config["train__status"] = "FINISHED"

        history = {k: v[-1] for k, v in history.items()}
        queue.put((history, arch_hash))

        log_train(csv_config, directory)

        logging.info("Finished train")

    except Exception as ex:
        if 'OOM' in str(ex):
            csv_config["train__status"] = "OOM ERROR"
        else:
            csv_config["train__status"] = "ERROR: " + str(ex)

        logging.error("Error: %s" % ex)
        logging.error(traceback.format_exc())
        sys.stdout.flush()
        queue.put(None)

        log_train(csv_config, directory)


def curves_fsl_train(model_creator, config, ifold, queue, debug, directory, timeout):
    logging.info('Starting training (fold %d) %s' % (ifold, config))

    try:
        training_config = config['train']
        net_config = config['model']
        data_config = config['data']

        data_name = data_config['dataset_name']
        data_target = data_config['dataset_target']

        task = get_task(data_name, data_target, model_creator)

        csv_config = flat_dict(config.copy())
        csv_config['train__max_epochs'] = csv_config.pop('train__epochs')
        csv_config['train__fold'] = ifold
        nhash = confighash(csv_config, exclude=HASH_EXCLUDE)
        arch_hash = confighash(csv_config, exclude=HASH_EXCLUDE + ["train__fold"])
        csv_config['run_hash'] = nhash
        csv_config['arch_hash'] = arch_hash

        import os
        import tensorflow as tf
        from phm_framework import models
        from phm_framework.models.utils import AdditionalRULValidationSets
        from phm_framework.optimization import hyper_parameters as hp

        # prepare output directory
        if not os.path.exists(directory):
            os.makedirs(directory)

        log_csv = load_log(None, directory)
        if not isinstance(log_csv, bool):
            query = log_csv[log_csv.run_hash == nhash]
            if query.shape[0] > 0 and query.iloc[0].train__status == 'FINISHED':
                r = query.iloc[0]
                queue.put(({
                               'val_loss': [r.val_loss],
                               'test_loss': [r.test_loss]
                           }, arch_hash))
                return

        # data reading and prepare data generators
        logging.info("Reading data")
        ts_len = secure_decode(training_config, "ts_len", dtype=int, task=task)
        random_state = secure_decode(training_config, "random_state", dtype=int, task=task)
        conditioning = secure_decode(training_config, "conditioning", dtype=str, default=None, task=task)
        preprocess = hp.PREPROCESS[secure_decode(data_config, "preprocess", str, default='norm', task=task)]()

        sequencer = FSLSequence
        sets = load_train_generators(data_name,
                                     task_name=data_target,
                                     ts_len=ts_len, fold=ifold, num_folds=config['train']['num_folds'],
                                     preprocess=preprocess,
                                     normalize_output=False,
                                     filters={"data": "curves"},
                                     random_state=random_state,
                                     return_train_val=True,
                                     return_test=True,
                                     sequencer=sequencer)

        train_gen = sets['train']
        train_gen.batches_per_epoch = 100
        val_gen = sets['val']
        test_gen = sets['test']

        ds = datasets.Dataset(data_name)
        _task = ds['final_loss']
        _task.filters = {"data": "results"}

        (results,) = _task.load()

        nfeatures = 0

        stride = 1
        train_gen.stride = stride
        val_gen.stride = stride
        test_gen.stride = stride

        # batch size
        batch_size = secure_decode(training_config, "batch_size", dtype=int, task=task)
        train_gen.batch_size = batch_size
        val_gen.batch_size = 256
        test_gen.batch_size = 256

        extra_channel = getattr(importlib.import_module(model_creator.__module__), 'EXTRA_CHANNEL')
        train_gen.extra_channel = extra_channel
        val_gen.extra_channel = extra_channel
        test_gen.extra_channel = extra_channel

        if "train_generator" in config:
            for key, value in config["train_generator"].items():
                setattr(train_gen, key, value)
        if "val_generator" in config:
            for key, value in config["val_generator"].items():
                setattr(val_gen, key, value)

        input_shape = (max(ts_len, 20), 3)

        logging.info("Finished Data reading")

        # training config
        epochs = secure_decode(training_config, "epochs", int, task=task)
        epochs = min(5, epochs) if debug else epochs

        lr = secure_decode(training_config, "lr", float, pop=False, task=task)
        monitor = secure_decode(training_config, "monitor", str, default="val_loss", task=task)
        verbose = secure_decode(training_config, "verbose", bool, default=False, task=task)

        output_dim = hp.get_output_dim(task)
        output = hp.get_output(task)

        # create and compile model

        model_params = models.get_model_params(net_config, model_creator, task)
        csv_config.update(flat_dict({'model': model_params}))
        csv_config['model__output'] = output
        csv_config['model__output_dim'] = output_dim
        csv_config['model__input_shape'] = input_shape
        if 'num_features' in model_params:
            csv_config['model__num_features'] = nfeatures
            model_params['num_features'] = nfeatures
        del model_params['output_dim']
        del model_params['input_shape']
        del model_params['output']
        model = model_creator(input_shape, output_dim=output_dim, output=output,
                              **model_params)
        logging.info("Model created")
        model.summary(print_fn=lambda x: logging.info(x))

        model.compile(loss='mse',
                      metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                               tf.keras.metrics.MeanAbsoluteError(name="mae")],
                      optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      run_eagerly=False)

        # train
        es = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=8)
        rlr = tf.keras.callbacks.ReduceLROnPlateau(patience=3)

        extra_callbacks = []

        logging.info("Started training")
        start_time = time.time()
        train_keys = train_gen.data.keys()

        history = model.fit(train_gen, validation_data=val_gen,
                            batch_size=batch_size,
                            epochs=epochs, verbose=(2 if verbose else 0),
                            callbacks=[es, rlr] + extra_callbacks)
        history = history.history

        # save csv opt_history
        csv_config['train__time'] = (time.time() - start_time)
        csv_config.update({k: history[k][-1] for k in history.keys() if k.startswith('val')})

        logging.info("Evaluating on test set")
        test_metrics = model.evaluate(test_gen, verbose=(2 if verbose else 0))
        for i, metric_name in enumerate(model.metrics_names):
            csv_config[f"test_{metric_name}"] = test_metrics[i]

        # simulate bayes optimization
        train_times = results[['unit', 'train__time']].set_index('unit').to_dict()['train__time']
        train_times = {tuple(k): train_times[k] for k in test_gen.units if k in train_times}

        assert len(train_times) > 0

        # Prepare validation data to generate decision rules
        X, Y, _ = discretize_data(model, results, train_gen, ts_len, val_gen, debug=debug)
        del X['unit']
        Y = Y[~X.best_performance.isnull()]
        X = X[~X.best_performance.isnull()]
        del X['best_performance']

        # Prepare simulation test data
        Xtest, Ytest, test_data = discretize_data(model, results, train_gen, ts_len, test_gen, debug=debug)

        former_tacc = -1
        for negative_class_threshold in np.arange(0.45, 1.0, 0.05):

            logging.info(f"Using negative_class_threshold = {negative_class_threshold}")
            csv_config['negative_class_threshold'] = negative_class_threshold

            nhash = confighash(csv_config, exclude=HASH_EXCLUDE)
            csv_config['run_hash'] = nhash

            # prepare output directory
            if not os.path.exists(directory):
                os.makedirs(directory)

            log_csv = load_log(None, directory)
            if not isinstance(log_csv, bool):
                query = log_csv[log_csv.run_hash == nhash]
                if query.shape[0] > 0 and query.iloc[0].train__status == 'FINISHED':
                    continue

            # Generate decision rules
            best_params, clf, tacc = generate_rule_tree(X, Y, negative_class_threshold)

            if tacc == 1:
                logging.info("Stop training")
                break
            elif tacc == former_tacc:
                logging.info("Skipping because same as former")
                continue
            else:
                former_tacc = tacc

            best_params['min_samples_leaf'] = 1000
            save_tree(arch_hash, clf, directory, best_params, nhash)

            experiments = set(['_'.join(e[0].split('_')[:-1]) for e in test_data])

            epochs_avoided = 0
            num_runs = 0
            num_prunings = 0
            total_epochs = 0
            total_train_time = 0
            avoided_train_time = 0
            filter_best_losses = []
            real_best_losses = []

            Xtest['pred'] = clf.predict(
                Xtest[['epoch', 'expected_improvement', 'val_improvement', 'prediction_uncertainty']])
            for experiment_id in tqdm.tqdm(experiments):

                # filter experiments runs
                eresults = results[results.unit.map(lambda x: experiment_id in x)]
                unit_ordered = list(eresults.unit)
                Xexp = Xtest[Xtest.unit.map(lambda x: x in unit_ordered)]
                unit_ordered = [u for u in unit_ordered if u in Xexp.unit.values]
                eresults = eresults[eresults.unit.map(lambda x: x in unit_ordered)]
                eresults = eresults[~eresults.train__time.isnull()]

                real_best_loss = Xexp.best_performance.min()

                real_epochs = pd.DataFrame([{'unit': d[0], 'epochs': d[-2].shape[0], 'final_val_loss': d[-2][-1][1]}
                                            for d in test_data if d[0] in unit_ordered])
                Xexp = pd.merge(Xexp, real_epochs, on='unit')
                filter_best_loss = Xexp[Xexp.unit == unit_ordered[0]].final_val_loss.iloc[0]

                for unit in unit_ordered[1:]:
                    decision_data = Xexp[Xexp.unit == unit]
                    # preds = clf.predict(decision_data[['epoch', 'expected_improvement', 'val_improvement',
                    #                                   'prediction_uncertainty']])
                    preds = decision_data.pred.values
                    # preds |= True

                    arr = np.asarray(preds, dtype=bool)  # Asegura que sea un array booleano
                    epochs = (np.argmax(~arr) if not arr.all() else len(arr)) + 1
                    ureal_epochs = decision_data.epochs.iloc[-1]
                    total_epochs += ureal_epochs
                    run_all = np.all(arr)
                    uepochs_avoided = 0 if run_all else ureal_epochs - epochs
                    epochs_avoided += uepochs_avoided
                    train_time = eresults[eresults.unit == unit].train__time.iloc[0]
                    total_train_time += train_time
                    avoided_train_time += ((train_time / ureal_epochs) * uepochs_avoided)
                    num_prunings += 0 if run_all else 1

                    if run_all and decision_data.final_val_loss.iloc[-1] < filter_best_loss:
                        filter_best_loss = decision_data.final_val_loss.iloc[-1]

                    num_runs += 1

                filter_best_losses.append(filter_best_loss)
                real_best_losses.append(real_best_loss)

            csv_config["epochs_avoided"] = epochs_avoided
            csv_config["total_epochs"] = total_epochs
            csv_config["total_train_time"] = total_train_time
            csv_config["avoided_train_time"] = avoided_train_time
            csv_config["filter_best_losses"] = filter_best_losses
            csv_config["real_best_losses"] = real_best_losses
            csv_config["num_prunings"] = num_prunings
            csv_config["num_runs"] = num_runs

            logging.info(f"epochs_avoided: {epochs_avoided}")
            logging.info(f"total_train_time: {total_train_time}")
            logging.info(f"avoided_train_time: {avoided_train_time}")
            logging.info(f"filter_best_losses: {filter_best_losses}")
            logging.info(f"real_best_losses: {real_best_losses}")
            logging.info(
                f"fount best: {np.mean([(np.abs(f - r) < 1e-06) or (f < r) for f, r in zip(filter_best_losses, real_best_losses)])}")
            logging.info(f"num_prunings: {num_prunings}")
            logging.info(f"num_runs: {num_runs}")


            csv_config["train__status"] = "FINISHED"

            log_train(copy.deepcopy(csv_config), directory)
            del csv_config["epochs_avoided"]
            del csv_config["total_epochs"]
            del csv_config["total_train_time"]
            del csv_config["avoided_train_time"]
            del csv_config["filter_best_losses"]
            del csv_config["real_best_losses"]
            del csv_config["num_prunings"]
            del csv_config["num_runs"]

        history = {k: v[-1] for k, v in history.items()}
        queue.put((history, arch_hash))

        logging.info("Finished train")

    except Exception as ex:
        if 'OOM' in str(ex):
            csv_config["train__status"] = "OOM ERROR"
        else:
            csv_config["train__status"] = "ERROR: " + str(ex)

        logging.error("Error: %s" % ex)
        logging.error(traceback.format_exc())
        sys.stdout.flush()
        queue.put(None)

        log_train(csv_config, directory)


def curves_fsl_train_v2(model_creator, config, ifold, queue, debug, directory, timeout):
    logging.info('Starting training (fold %d) %s' % (ifold, config))

    try:
        training_config = config['train']
        net_config = config['model']
        data_config = config['data']

        data_name = data_config['dataset_name']
        data_target = data_config['dataset_target']

        task = get_task(data_name, data_target, model_creator)

        csv_config = flat_dict(config.copy())
        csv_config['train__max_epochs'] = csv_config.pop('train__epochs')
        csv_config['train__fold'] = ifold
        nhash = confighash(csv_config, exclude=HASH_EXCLUDE)
        arch_hash = confighash(csv_config, exclude=HASH_EXCLUDE + ["train__fold"])
        csv_config['arch_hash'] = arch_hash

        import os
        import tensorflow as tf
        from phm_framework import models
        from phm_framework.models.utils import AdditionalRULValidationSets
        from phm_framework.optimization import hyper_parameters as hp

        # prepare output directory
        if not os.path.exists(directory):
            os.makedirs(directory)


        # data reading and prepare data generators
        logging.info("Reading data")
        ts_len = secure_decode(training_config, "ts_len", dtype=int, task=task)
        random_state = secure_decode(training_config, "random_state", dtype=int, task=task)

        if net_config['net'] == 'rnn':
            sequencer = SequenceV2
        else:
            sequencer = FSLSequence
        sets = load_train_net_generators_v2(data_name,
                                            task_name=data_target,
                                            ts_len=ts_len, fold=ifold, num_folds=config['train']['num_folds'],
                                            normalize_output=False,
                                            filters={"data": "curves"},
                                            test_dataset_names=data_config['test_dataset_names'],
                                            random_state=random_state,
                                            sequencer=sequencer)

        train_gen = sets['train']
        train_gen.batches_per_epoch = 100
        val_gen = sets['val']
        test_gen = sets['test']

        ds = datasets.Dataset(data_name)
        _task = ds['final_loss']
        _task.filters = {"data": "results"}

        (results,) = _task.load()

        nfeatures = 0

        stride = 1
        train_gen.stride = stride
        val_gen.stride = stride
        test_gen.stride = stride

        # batch size
        batch_size = secure_decode(training_config, "batch_size", dtype=int, task=task)
        train_gen.batch_size = batch_size
        val_gen.batch_size = 256

        extra_channel = getattr(importlib.import_module(model_creator.__module__), 'EXTRA_CHANNEL')
        train_gen.extra_channel = extra_channel
        val_gen.extra_channel = extra_channel

        if "train_generator" in config:
            for key, value in config["train_generator"].items():
                setattr(train_gen, key, value)
        if "val_generator" in config:
            for key, value in config["val_generator"].items():
                setattr(val_gen, key, value)

        input_shape = (max(ts_len, 20), 3)

        logging.info("Finished Data reading")

        # training config
        epochs = secure_decode(training_config, "epochs", int, task=task)

        lr = secure_decode(training_config, "lr", float, pop=False, task=task)
        monitor = secure_decode(training_config, "monitor", str, default="val_loss", task=task)
        verbose = secure_decode(training_config, "verbose", bool, default=False, task=task)

        output_dim = hp.get_output_dim(task)
        output = hp.get_output(task)

        # create and compile model

        model_params = models.get_model_params(net_config, model_creator, task)
        csv_config.update(flat_dict({'model': model_params}))
        csv_config['model__output'] = output
        csv_config['model__output_dim'] = output_dim
        csv_config['model__input_shape'] = input_shape
        if 'num_features' in model_params:
            csv_config['model__num_features'] = nfeatures
            model_params['num_features'] = nfeatures
        del model_params['output_dim']
        del model_params['input_shape']
        del model_params['output']
        model = model_creator(input_shape, output_dim=output_dim, output=output,
                              **model_params)
        logging.info("Model created")
        model.summary(print_fn=lambda x: logging.info(x))

        model.compile(loss='mse',
                      metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                               tf.keras.metrics.MeanAbsoluteError(name="mae")],
                      optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      run_eagerly=False)

        # train
        es = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=8)
        rlr = tf.keras.callbacks.ReduceLROnPlateau(patience=3)

        extra_callbacks = []

        logging.info("Started training")

        history = model.fit(train_gen, validation_data=val_gen,
                            batch_size=batch_size,
                            epochs=epochs, verbose=(2 if verbose else 0),
                            callbacks=[es, rlr] + extra_callbacks)
        history = history.history

        # discretize data
        Xtest, Ytest, curves = extended_decision_data(model, results, train_gen, ts_len, test_gen, debug=debug)
        Xtest['continue'] = Ytest

        # save test estimations
        data_dir = os.path.join(directory, 'data')
        if not os.path.exists(data_dir): os.makedirs(data_dir)
        data_file = os.path.join(data_dir, f'test_{nhash}.data')
        pk.dump((Xtest, curves), open(data_file, 'wb'))

        queue.put((history, arch_hash, data_file, csv_config))

        #logging.info(f"Finished training fold {ifold} of {arch_hash}")

    except Exception as ex:
        if 'OOM' in str(ex):
            csv_config["train__status"] = "OOM ERROR"
        else:
            csv_config["train__status"] = "ERROR: " + str(ex)

        logging.error("Error: %s" % ex)
        logging.error(traceback.format_exc())
        sys.stdout.flush()
        queue.put(None)

        log_train(csv_config, directory)


def find_optimal_strategy_tree(X_train, Y_train, X_val, curves, opt_history, directory):
    """
    Busca el árbol que maximiza el éxito de la estrategia global.
    """
    best_score = -np.inf
    best_tree = None
    best_params = None

    print(f"ACC(T)\tACC(F)\tScore\tPerf.Score\tT.Score")
    print(f"------\t------\t-----\t----------\t-------")

    rules_founds = []
    for negative_class_threshold in np.arange(0.45, 1.0, 0.05):
        # Espacio de búsqueda de configuraciones de árbol
        for max_depth in [2, 3, 4, 5]:

            for min_samples in [50, 100, 200]:

                for w in range(10, 100, 10):

                    # 1. Entrenar un árbol candidato con los datos de todos los folds
                    params = {}
                    params['min_samples'] = min_samples
                    params['max_depth'] = max_depth
                    params['continue_weight'] = w
                    candidate_tree, tacc, facc = train_rule_tree(X_train, Y_train, params, negative_class_threshold)

                    # avoided previously run simulations
                    rules = export_text(candidate_tree, feature_names=candidate_tree.feature_names_in_)
                    if rules in rules_founds:
                        continue
                    rules_founds.append(rules)

                    # 2. SIMULAR la estrategia en todos los folds
                    current_strategy_score, performance_score, time_score = simulate_strategy(X_val, curves, opt_history, candidate_tree)

                    print(
                        f" {tacc:0.2f}\t {facc:0.2f}\t{current_strategy_score:0.3f}\t{performance_score:0.3f}\t{time_score:0.3f}", end="")
                    # 3. Guardar el mejor
                    if current_strategy_score > best_score:
                        best_score = current_strategy_score
                        best_tree = candidate_tree
                        best_params = copy.deepcopy(params)
                        print("*")
                    else:
                        print("")



    #save_tree(X_val, arch_hash, best_tree, directory, nhash, best_params)

    return best_tree, best_params, best_score


def simulate_strategy(Xtest, curves, opt_history, clf, csv_config=None):
    Xtest = Xtest.copy()

    epochs_avoided = 0
    num_runs = 0
    num_prunings = 0
    total_epochs = 0
    total_train_time = 0
    avoided_train_time = 0
    filter_best_losses = []
    real_best_losses = []
    rank_losses = []

    Xtest['pred'] = clf.predict(Xtest[clf.feature_names_in_])

    # select experiments
    experiments_in_test = Xtest.unit.map(lambda x: "_".join(x.split("_")[:-1])).unique()
    experiments_in_curves = set(['_'.join(e[0].split('_')[:-1]) for e in curves])
    experiments = [e for e in experiments_in_test if e in experiments_in_curves]

    for experiment_id in experiments:

        # filter experiments runs
        eresults = opt_history[opt_history.unit.map(lambda x: experiment_id in x)]
        unit_ordered = list(eresults.unit)
        Xexp = Xtest[Xtest.unit.map(lambda x: x in unit_ordered)]
        unit_ordered = [u for u in unit_ordered if u in Xexp.unit.values]
        eresults = eresults[eresults.unit.map(lambda x: x in unit_ordered)]
        eresults = eresults[~eresults.train__time.isnull()]

        real_best_loss = Xexp.best_performance.min()

        real_epochs = pd.DataFrame([{'unit': d[0], 'epochs': d[-2].shape[0], 'final_val_loss': d[-2][-1][1]}
                                    for d in curves if d[0] in unit_ordered])
        Xexp = pd.merge(Xexp, real_epochs, on='unit')
        filter_best_loss = Xexp[Xexp.unit == unit_ordered[0]].final_val_loss.iloc[0]

        for unit in unit_ordered[1:]:
            decision_data = Xexp[Xexp.unit == unit].sort_values('epoch')
            preds = decision_data.pred.values  # Predicciones crudas del modelo

            # Configuración de robustez
            patience = 3  # Épocas consecutivas de "Parar" necesarias
            stop_counter = 0
            final_stop_epoch = len(preds)  # Por defecto, llega al final
            is_pruned = False

            for i, p in enumerate(preds):
                if not p:  # El modelo sugiere PARAR
                    stop_counter += 1
                else:
                    stop_counter = 0  # Reset si hay una señal de "Continuar"

                if stop_counter >= patience:
                    final_stop_epoch = i + 1
                    is_pruned = True
                    break  # Una vez parado, no se evalúan más épocas (Monotonicidad)

                # Cálculo de métricas basado en la decisión final consolidada
            ureal_epochs = decision_data.epochs.iloc[-1]
            total_epochs += ureal_epochs

            epochs_to_run = final_stop_epoch if is_pruned else ureal_epochs
            uepochs_avoided = ureal_epochs - epochs_to_run

            run_all = uepochs_avoided == 0

            epochs_avoided += uepochs_avoided
            train_time = eresults[eresults.unit == unit].train__time.iloc[0]
            total_train_time += train_time
            avoided_train_time += ((train_time / ureal_epochs) * uepochs_avoided)
            num_prunings += 0 if run_all else 1

            if run_all and decision_data.final_val_loss.iloc[-1] < filter_best_loss:
                filter_best_loss = decision_data.final_val_loss.iloc[-1]

            num_runs += 1

        filter_best_losses.append(filter_best_loss)
        real_best_losses.append(real_best_loss)
        rank_losses.append(sorted(Xexp.groupby('unit').final_val_loss.max().values).index(filter_best_loss))

    performance_score = np.mean([r / f for r, f in zip(real_best_losses, filter_best_losses)])
    time_score = (epochs_avoided / total_epochs)

    score = (0.5 * performance_score + 0.5 * time_score)

    if csv_config:
        csv_config["epochs_avoided"] = epochs_avoided
        csv_config["total_epochs"] = total_epochs
        csv_config["total_train_time"] = total_train_time
        csv_config["avoided_train_time"] = avoided_train_time
        csv_config["filter_best_losses"] = filter_best_losses
        csv_config["real_best_losses"] = real_best_losses
        csv_config["rank_losses"] = rank_losses
        csv_config["num_prunings"] = num_prunings
        csv_config["num_runs"] = num_runs

        logging.info(f"epochs_avoided: {epochs_avoided}")
        logging.info(f"total_train_time: {total_train_time}")
        logging.info(f"avoided_train_time: {avoided_train_time}")
        logging.info(f"filter_best_losses: {filter_best_losses}")
        logging.info(f"real_best_losses: {real_best_losses}")
        logging.info(
            f"fount best: {np.mean([(np.abs(f - r) < 1e-06) or (f < r) for f, r in zip(filter_best_losses, real_best_losses)])}")
        logging.info(f"num_prunings: {num_prunings}")
        logging.info(f"num_runs: {num_runs}")

        csv_config["train__status"] = "FINISHED"
        csv_config["score"] = score

        return csv_config

    else:

        return score, performance_score, time_score


def generate_rule_tree(X, Y, negative_class_threshold=0.5):
    def score(estimator, X, y):
        p = estimator.predict(X)
        return ((2 / 5) * (p == y).values[np.where(~y)].mean() + (3 / 5) * (p == y).values[np.where(y)].mean())

    best_score = 0
    best_params = None
    for d in range(2, 10):
        for w in range(10, 100, 10):
            s = model_selection.cross_val_score(tree.DecisionTreeClassifier(max_depth=d,
                                                                            min_samples_leaf=1000,
                                                                            class_weight={False: 1, True: w}),
                                                X, y=Y, scoring=score).mean()

            if s > best_score:
                best_score = s
                best_params = {'max_depth': d,
                               'continue_weight': w}
                logging.info(f"{str(best_params)}, {s}")

    return train_rule_tree(X, Y, best_params, negative_class_threshold)




def discretize_data(model, results, support_gen, ts_len, data_gen, debug=False):
    X, Y, data = prepare_decision_data(model, results, support_gen, ts_len, data_gen, debug)

    X['expected_improvement'] = pd.cut(X['expected_improvement'], 5, labels=range(5))
    X['val_improvement'] = pd.cut(X['val_improvement'], 5, labels=range(5))
    X['prediction_uncertainty'] = pd.cut(X['prediction_uncertainty'], 3, labels=range(3))

    return X, Y, data

def prepare_decision_data(model, results, support_gen, ts_len, data_gen, debug=False):
    data = get_curve_predictions_reusing(model, support_gen, ts_len, data_gen, results, debug=debug)
    experiments = set(['_'.join(e[0].split('_')[:-1]) for e in data])
    ordered_experiments = list(results.unit)
    ext_data = []
    for experiment in experiments:
        edata = list(filter(lambda i: experiment in i[0], data))
        edata = list(filter(lambda i: i[0] in ordered_experiments, edata))
        edata = sorted(edata, key=lambda x: ordered_experiments.index(x[0]))

        best = edata[0][-1]

        r = edata[0]
        ext_data.append({
            'epoch': -1,
            'unit': r[0],
            'final_performance': r[-1],
            'best_performance': None,
            'val_performance': None,
            'train_performance': None,
            'continue': True,
            'predicted_performance': None,
            'prediction_uncertainty': None
        })

        for r in edata[1:]:
            continue_run = r[-1] < best

            for e in range(len(r[1])):
                ext_data.append({
                    'epoch': e,
                    'unit': r[0],
                    'final_performance': r[-1],
                    'best_performance': best,
                    'val_performance': r[1][-1][e][1],
                    'train_performance': r[1][-1][e][0],
                    'continue': continue_run,
                    'predicted_performance': r[2][e],
                    'prediction_uncertainty': r[3][e]
                })

            if continue_run:
                best = r[-1]

    # Create decisión tree rule
    X = pd.DataFrame(ext_data)
    Y = X['continue']

    X['expected_improvement'] = ((X.predicted_performance - X.best_performance) / X.best_performance).clip(-1, 1)

    X['val_improvement'] = ((X.val_performance - X.best_performance) / X.best_performance).clip(-1, 1)

    X = X[['unit', 'epoch', 'best_performance', 'expected_improvement', 'val_improvement',
           'prediction_uncertainty', 'val_performance', 'predicted_performance']]

    return X, Y, data


def extended_decision_data(model, results, support_gen, ts_len, data_gen, debug=False):
    # ... (mantenemos la lógica inicial para obtener ext_data) ...
    X, Y, data = prepare_decision_data(model, results, support_gen, ts_len, data_gen, debug=False)
    X['continue'] = Y

    # 1. Velocidad de mejora: Diferencia entre la época actual y la anterior
    X['val_velocity'] = X.groupby('unit')['val_performance'].diff().fillna(0)

    # 2. Suavizado EMA: Para ignorar picos de ruido en la validación
    X['val_ema'] = X.groupby('unit')['val_performance'].transform(lambda x: x.ewm(span=3).mean())

    # Feature set ampliado
    logging.info(f"Before remove rows with nulls. Shape: {X.shape}")

    features = ['unit', 'epoch', 'expected_improvement', 'val_improvement',
                'prediction_uncertainty', 'val_velocity', 'val_ema', 'best_performance',
                'continue']
    X = X[features]
    X = X[~X.T.isnull().any()]

    Y = X['continue']
    del X['continue']

    logging.info(f"Finalized data discretization. Shape: {X.shape}")

    return X, Y, data


def save_tree(arch_hash, clf, directory, params, nhash=None):
    if nhash:
        suffix = f"{arch_hash}_{nhash}"
    else:
        suffix = arch_hash

    tree.plot_tree(clf, feature_names=clf.feature_names_in_)
    if not os.path.exists(os.path.join(directory, 'trees')):
        os.makedirs(os.path.join(directory, 'trees'))

    pk.dump(clf, open(os.path.join(directory, 'trees', f'tree_{suffix}.pk'), 'wb'))
    pk.dump(params, open(os.path.join(directory, 'trees', f'tree_params_{suffix}.pk'), 'wb'))
    plt.savefig(os.path.join(directory, 'trees', f'tree_{suffix}.svg'))


def get_curve_predictions(model, support_gen, ts_len, data_gen, results, debug=False):
    input_len = max(ts_len, 20)
    (_, supports, sys) = zip(*[support_gen[i][0] for i in range(3)])
    N = len(data_gen.data.items())
    n = N // 10 if debug else N

    aux_inputs = []
    units = []
    raw_signals = []
    final_performances = []

    for unit, signal in tqdm.tqdm(list(data_gen.data.items())[:n]):
        signals = []
        for i in range(1, min(ts_len, signal.shape[0])):
            if i < ts_len:
                mask = np.vstack((np.ones((i, 1)), np.vstack(np.zeros((input_len - i, 1)))))
                s = np.vstack((signal[:i], np.vstack(np.zeros((input_len - i, 2)))))
            else:
                mask = np.ones((input_len, 1))
                s = signal

            s = np.hstack((s, mask))
            signals.append(s)

        signals = np.array(signals)

        aux_inputs.append(signals)
        units.append(unit)
        raw_signals.append(signal)
        final_performances.append(signal[-1][1])

    preds = []
    for support, sy in tqdm.tqdm(zip(supports, sys)):
        preds_ = model((np.vstack(aux_inputs), support, sy)).numpy()
        preds.append(preds_)

    preds, stds = np.mean(preds, axis=0), np.std(preds, axis=0)

    data = []
    i = 0
    for unit, final_performance, signal, signals in zip(units, final_performances, raw_signals, aux_inputs):
        j = i + len(signals)
        data.append((''.join(unit), signals, preds[i:i + j], stds[i:i + j], signal, signal[-1][1]))

    return data


def get_curve_predictions_reusing(model, support_gen, ts_len, data_gen, opt_history, debug=False):
    input_len = max(ts_len, 20)
    (_, supports, sys) = zip(*[support_gen[i][0] for i in range(3)])
    supports_cache = np.copy(supports)
    N = len(data_gen.data.items())
    n = N // 10 if debug else N

    aux_inputs = []
    units = []
    raw_signals = []
    final_performances = []
    preds = []

    sorted_units = list(opt_history.unit.values)
    curves = list(data_gen.data.items())
    curves = list(filter(lambda x: ''.join(x[0]) in sorted_units, curves))
    curves = sorted(curves, key=lambda x: sorted_units.index(''.join(x[0])))

    j = 0
    current_experiment = None
    for k, (unit, signal) in enumerate(tqdm.tqdm(curves[:n])):
        signals = []
        y = signal[-1, 1]
        experiment = ''.join(unit).split("_")[:-1]

        for i in range(1, min(ts_len, signal.shape[0])):
            if i < ts_len:
                mask = np.vstack((np.ones((i, 1)), np.vstack(np.zeros((input_len - i, 1)))))
                s = np.vstack((signal[:i], np.vstack(np.zeros((input_len - i, 2)))))
            else:
                mask = np.ones((input_len, 1))
                s = signal

            s = np.hstack((s, mask))
            signals.append(s)

        signals = np.array(signals)
        for support, sy in zip(supports, sys):
            preds_ = model((signals, support, sy)).numpy()
            preds.append(preds_)

        signal = signal[:input_len]
        i = signal.shape[0]
        if i < input_len:
            mask = np.vstack((np.ones((i, 1)), np.vstack(np.zeros((input_len - i, 1)))))
            s = np.vstack((signal[:i], np.vstack(np.zeros((input_len - i, 2)))))
        else:
            mask = np.ones((i, 1))
            s = signal

        s = np.hstack((s, mask))

        if experiment != current_experiment:
            supports = np.copy(supports_cache)
            current_experiment = experiment
            supports[j % 3][k % 100] = s
            sys[j % 3][k % 100] = y

        aux_inputs.append(signals)
        units.append(unit)
        raw_signals.append(signal)
        final_performances.append(signal[-1][1])

    preds = np.hstack([preds[i:i + 3] for i in range(0, len(preds), 3)])
    preds, stds = np.mean(preds, axis=0), np.std(preds, axis=0)

    data = []
    i = 0
    for unit, final_performance, signal, signals in zip(units, final_performances, raw_signals, aux_inputs):
        j = i + len(signals)
        data.append((''.join(unit), signals, preds[i:i + j], stds[i:i + j], signal, signal[-1][1]))

    return data


def arima_train(model_creator, config, ifold, queue, debug, directory, timeout):
    logging.info('Starting training (fold %d) %s' % (ifold, config))

    try:
        training_config = config['train']
        net_config = config['model']
        data_config = config['data']

        data_name = data_config['dataset_name']
        data_target = data_config['dataset_target']

        task = get_task(data_name, data_target, model_creator)

        csv_config = flat_dict(config.copy())
        csv_config['train__fold'] = ifold
        nhash = confighash(csv_config, exclude=HASH_EXCLUDE)
        arch_hash = confighash(csv_config, exclude=HASH_EXCLUDE + ["train__fold"])
        csv_config['run_hash'] = nhash
        csv_config['arch_hash'] = arch_hash

        import os
        import tensorflow as tf
        from phm_framework import models
        from phm_framework.models.utils import AdditionalRULValidationSets
        from phm_framework.optimization import hyper_parameters as hp

        # prepare output directory
        if not os.path.exists(directory):
            os.makedirs(directory)

        log_csv = load_log(None, directory)
        if not isinstance(log_csv, bool):
            query = log_csv[log_csv.run_hash == nhash]
            if query.shape[0] > 0 and query.iloc[0].train__status == 'FINISHED':
                r = query.iloc[0]
                queue.put(({
                               'test_mse': [r.test_mse]
                           }, arch_hash))
                return

        # data reading and prepare data generators
        logging.info("Reading data")
        ts_len = secure_decode(training_config, "ts_len", dtype=int, task=task)
        random_state = secure_decode(training_config, "random_state", dtype=int, task=task)
        preprocess = hp.PREPROCESS[secure_decode(data_config, "preprocess", str, default='norm', task=task)]()

        sequencer = Sequence
        sets = load_train_generators(data_name,
                                     task_name=data_target,
                                     ts_len=ts_len, fold=ifold, num_folds=config['train']['num_folds'],
                                     preprocess=preprocess, return_test=True,
                                     normalize_output=False,
                                     filters={"data": "curves"},
                                     random_state=random_state,
                                     return_train_val=False,
                                     sequencer=sequencer)

        test_gen = sets['test']

        ds = datasets.Dataset(data_name)
        _task = ds['final_loss']
        _task.filters = {"data": "results"}

        (results,) = _task.load()
        input_shape = (ts_len, 2)
        logging.info("Finished Data reading")

        verbose = secure_decode(training_config, "verbose", bool, default=False, task=task)

        # create and compile model
        csv_config['model__input_shape'] = input_shape

        logging.info("Started training")

        # simulate bayes optimization
        train_times = results[['unit', 'train__time']].set_index('unit').to_dict()['train__time']
        train_times = {tuple(k): train_times[k] for k in test_gen.units if k in train_times}

        assert len(train_times) > 0

        units, inputs, signals = zip(*[(u, v[:ts_len, :], v) for u, v in test_gen.data.items()])

        aux = {u: (s, train_times[u]) for u, s in zip(units, signals) if u in train_times}
        test_datasets = set([u.split('_')[0] for u in test_gen.units])

        results = results[results.dataset.map(lambda x: x in test_datasets)]

        epochs_avoided = 0
        num_runs = 0
        num_prunings = 0
        total_epochs = 0
        total_train_time = 0
        avoided_train_time = 0
        experiments = results[['data__dataset_name', 'data__dataset_target', 'model__net']].drop_duplicates().values
        filter_best_losses = []
        real_best_losses = []
        mses = []
        maes = []

        if debug:
            experiments = experiments[:1]

        with tqdm.tqdm(total=len(units)) as progress_bar:
            for dataset, task, net in experiments:
                unit_ordered = list(
                    results[(results.dataset == dataset) & (results.net == net) & (results.task == task)].unit)

                if debug:
                    unit_ordered = unit_ordered[:10]

                filter_best_loss = 100000
                real_best_loss = 10000

                for unit in unit_ordered:

                    if unit in test_gen.units:
                        unit_data = aux[tuple(unit)]
                        curve = unit_data[0][:, 1]
                        partial_curve = curve[:ts_len]
                        real_val_loss = curve[-1]
                        real_epochs = unit_data[0].shape[0]

                        best_params = None
                        best_mse = 1000000
                        pred_val_loss = None
                        best_mae = None
                        for p in range(1, 5 - 1):
                            for d in range(1, ts_len - 1):
                                for q in range(1, ts_len - 1):
                                    try:
                                        model = ARIMA(partial_curve, order=(p, d, q))
                                        mfit = model.fit()

                                        if mfit.mse < best_mse:
                                            best_params = (p, d, q)
                                            pred_val_loss = mfit.forecast(100 - ts_len)[-1]
                                            best_mse = mfit.mse
                                            best_mae = mfit.mae
                                    except:
                                        pass

                        mses.append(best_mse)
                        maes.append(best_mae)

                        train_time = unit_data[1]
                        total_train_time += train_time
                        total_epochs += real_epochs
                        num_runs += 1

                        if real_val_loss < real_best_loss:  # real best loss found during real training
                            real_best_loss = real_val_loss

                        if pred_val_loss < filter_best_loss * 2:  # no stopped
                            if real_val_loss < filter_best_loss:  # best loss found during simulated training
                                filter_best_loss = real_val_loss

                        else:
                            epochs_avoided += (real_epochs - ts_len)
                            avoided_train_time += ((train_time / real_epochs) * (real_epochs - ts_len))
                            num_prunings += 1

                    progress_bar.update(1)

                filter_best_losses.append(filter_best_loss)
                real_best_losses.append(real_best_loss)

        csv_config["epochs_avoided"] = epochs_avoided
        csv_config["total_epochs"] = total_epochs
        csv_config["total_train_time"] = total_train_time
        csv_config["avoided_train_time"] = avoided_train_time
        csv_config["filter_best_losses"] = filter_best_losses
        csv_config["real_best_losses"] = real_best_losses
        csv_config["num_prunings"] = num_prunings
        csv_config["num_runs"] = num_runs
        csv_config["test_mse"] = np.mean(mses)
        csv_config["test_mae"] = np.mean(maes)

        logging.info(f"epochs_avoided: {epochs_avoided}")
        logging.info(f"total_train_time: {total_train_time}")
        logging.info(f"avoided_train_time: {avoided_train_time}")
        logging.info(f"filter_best_losses: {filter_best_losses}")
        logging.info(f"real_best_losses: {real_best_losses}")
        logging.info(
            f"fount best: {np.mean([np.abs(f - r) < 1e-06 for f, r in zip(filter_best_losses, real_best_losses)])}")
        logging.info(f"num_prunings: {num_prunings}")
        logging.info(f"num_runs: {num_runs}")

        csv_config["train__status"] = "FINISHED"

        history = {"test_mse": [np.mean(mses)]}
        queue.put((history, arch_hash))

        log_train(csv_config, directory)

        logging.info("Finished train")

    except Exception as ex:
        if 'OOM' in str(ex):
            csv_config["train__status"] = "OOM ERROR"
        else:
            csv_config["train__status"] = "ERROR: " + str(ex)

        logging.error("Error: %s" % ex)
        logging.error(traceback.format_exc())
        sys.stdout.flush()
        queue.put(None)

        log_train(csv_config, directory)


def last_seen(model_creator, config, ifold, queue, debug, directory, timeout):
    logging.info('Starting training (fold %d) %s' % (ifold, config))

    try:
        training_config = config['train']
        data_config = config['data']

        data_name = data_config['dataset_name']
        data_target = data_config['dataset_target']

        task = get_task(data_name, data_target, model_creator)

        csv_config = flat_dict(config.copy())
        csv_config['train__fold'] = ifold
        nhash = confighash(csv_config, exclude=HASH_EXCLUDE)
        arch_hash = confighash(csv_config, exclude=HASH_EXCLUDE + ["train__fold"])
        csv_config['run_hash'] = nhash
        csv_config['arch_hash'] = arch_hash

        import os
        import tensorflow as tf
        from phm_framework import models
        from phm_framework.models.utils import AdditionalRULValidationSets
        from phm_framework.optimization import hyper_parameters as hp

        # prepare output directory
        if not os.path.exists(directory):
            os.makedirs(directory)

        log_csv = load_log(None, directory)
        if not isinstance(log_csv, bool):
            query = log_csv[log_csv.run_hash == nhash]
            if query.shape[0] > 0 and query.iloc[0].train__status == 'FINISHED':
                r = query.iloc[0]
                queue.put(({
                               'test_mse': [r.test_mse]
                           }, arch_hash))
                return

        # data reading and prepare data generators
        logging.info("Reading data")
        ts_len = secure_decode(training_config, "ts_len", dtype=int, task=task)
        random_state = secure_decode(training_config, "random_state", dtype=int, task=task)
        preprocess = hp.PREPROCESS[secure_decode(data_config, "preprocess", str, default='norm', task=task)]()

        sequencer = Sequence
        sets = load_train_generators(data_name,
                                     task_name=data_target,
                                     ts_len=ts_len, fold=ifold, num_folds=config['train']['num_folds'],
                                     preprocess=preprocess, return_test=True,
                                     normalize_output=False,
                                     filters={"data": "curves"},
                                     random_state=random_state,
                                     return_train_val=False,
                                     sequencer=sequencer)

        test_gen = sets['test']

        ds = datasets.Dataset(data_name)
        _task = ds['final_loss']
        _task.filters = {"data": "results"}

        (results,) = _task.load()

        input_shape = (ts_len, 2)
        logging.info("Finished Data reading")

        # create and compile model
        csv_config['model__input_shape'] = input_shape

        logging.info("Started training")

        # simulate bayes optimization
        train_times = results[['unit', 'train__time']].set_index('unit').to_dict()['train__time']
        train_times = {tuple(k): train_times[k] for k in test_gen.units if k in train_times}

        assert len(train_times) > 0

        units, inputs, signals = zip(*[(u, v[:ts_len, :], v) for u, v in test_gen.data.items()])

        aux = {u: (s, train_times[u]) for u, s in zip(units, signals) if u in train_times}
        test_datasets = set([u.split('_')[0] for u in test_gen.units])

        results = results[results.dataset.map(lambda x: x in test_datasets)]

        epochs_avoided = 0
        num_runs = 0
        num_prunings = 0
        total_epochs = 0
        total_train_time = 0
        avoided_train_time = 0
        experiments = results[['data__dataset_name', 'data__dataset_target', 'model__net']].drop_duplicates().values
        filter_best_losses = []
        real_best_losses = []

        if debug:
            experiments = experiments[:1]

        with tqdm.tqdm(total=len(units)) as progress_bar:
            for dataset, task, net in experiments:
                unit_ordered = list(
                    results[(results.dataset == dataset) & (results.net == net) & (results.task == task)].unit)

                if debug:
                    unit_ordered = unit_ordered[:10]

                filter_best_loss = 100000
                real_best_loss = 10000

                for unit in unit_ordered:

                    if unit in test_gen.units:
                        unit_data = aux[tuple(unit)]
                        curve = unit_data[0][:, 1]
                        partial_curve = curve[:ts_len]
                        real_val_loss = curve[-1]
                        real_epochs = unit_data[0].shape[0]

                        train_time = unit_data[1]
                        total_train_time += train_time
                        total_epochs += real_epochs
                        num_runs += 1

                        if real_val_loss < real_best_loss:  # real best loss found during real training
                            real_best_loss = real_val_loss

                        pred_val_loss = partial_curve[-1]
                        if pred_val_loss < filter_best_loss * 2:  # no stopped
                            if real_val_loss < filter_best_loss:  # best loss found during simulated training
                                filter_best_loss = real_val_loss

                        else:
                            epochs_avoided += (real_epochs - ts_len)
                            avoided_train_time += ((train_time / real_epochs) * (real_epochs - ts_len))
                            num_prunings += 1

                    progress_bar.update(1)

                filter_best_losses.append(filter_best_loss)
                real_best_losses.append(real_best_loss)

        csv_config["epochs_avoided"] = epochs_avoided
        csv_config["total_epochs"] = total_epochs
        csv_config["total_train_time"] = total_train_time
        csv_config["avoided_train_time"] = avoided_train_time
        csv_config["filter_best_losses"] = filter_best_losses
        csv_config["real_best_losses"] = real_best_losses
        csv_config["num_prunings"] = num_prunings
        csv_config["num_runs"] = num_runs

        csv_config["test_mse"] = 0

        logging.info(f"epochs_avoided: {epochs_avoided}")
        logging.info(f"total_train_time: {total_train_time}")
        logging.info(f"avoided_train_time: {avoided_train_time}")
        logging.info(f"filter_best_losses: {filter_best_losses}")
        logging.info(f"real_best_losses: {real_best_losses}")
        logging.info(
            f"fount best: {np.mean([np.abs(f - r) < 1e-06 for f, r in zip(filter_best_losses, real_best_losses)])}")
        logging.info(f"num_prunings: {num_prunings}")
        logging.info(f"num_runs: {num_runs}")

        csv_config["train__status"] = "FINISHED"

        history = {"test_mse": [0]}
        queue.put((history, arch_hash))

        log_train(csv_config, directory)

        logging.info("Finished train")

    except Exception as ex:
        if 'OOM' in str(ex):
            csv_config["train__status"] = "OOM ERROR"
        else:
            csv_config["train__status"] = "ERROR: " + str(ex)

        logging.error("Error: %s" % ex)
        logging.error(traceback.format_exc())
        sys.stdout.flush()
        queue.put(None)

        log_train(csv_config, directory)


def random_train(model_creator, config, ifold, queue, debug, directory, timeout):
    logging.info('Starting training (fold %d) %s' % (ifold, config))

    try:
        training_config = config['train']
        data_config = config['data']

        data_name = data_config['dataset_name']
        data_target = data_config['dataset_target']

        task = get_task(data_name, data_target, model_creator)

        csv_config = flat_dict(config.copy())
        csv_config['train__fold'] = ifold
        nhash = confighash(csv_config, exclude=HASH_EXCLUDE)
        arch_hash = confighash(csv_config, exclude=HASH_EXCLUDE + ["train__fold"])
        csv_config['run_hash'] = nhash
        csv_config['arch_hash'] = arch_hash

        import os
        import tensorflow as tf
        from phm_framework import models
        from phm_framework.models.utils import AdditionalRULValidationSets
        from phm_framework.optimization import hyper_parameters as hp

        # prepare output directory
        if not os.path.exists(directory):
            os.makedirs(directory)

        log_csv = load_log(None, directory)
        if not isinstance(log_csv, bool):
            query = log_csv[log_csv.run_hash == nhash]
            if query.shape[0] > 0 and query.iloc[0].train__status == 'FINISHED':
                r = query.iloc[0]
                queue.put(({
                               'test_mse': [r.test_mse]
                           }, arch_hash))
                return

        # data reading and prepare data generators
        logging.info("Reading data")
        random_pct = secure_decode(training_config, "random_pct", dtype=float, task=task)
        random_state = secure_decode(training_config, "random_state", dtype=int, task=task)

        random.seed(random_state)

        preprocess = hp.PREPROCESS[secure_decode(data_config, "preprocess", str, default='norm', task=task)]()

        sequencer = Sequence
        sets = load_train_generators(data_name,
                                     task_name=data_target,
                                     ts_len=2, fold=ifold, num_folds=config['train']['num_folds'],
                                     preprocess=preprocess, return_test=True,
                                     normalize_output=False,
                                     filters={"data": "curves"},
                                     random_state=random_state,
                                     return_train_val=False,
                                     sequencer=sequencer)

        test_gen = sets['test']

        ds = datasets.Dataset(data_name)
        _task = ds['final_loss']
        _task.filters = {"data": "results"}

        (results,) = _task.load()
        logging.info("Finished Data reading")

        # create and compile model

        logging.info("Started training")

        # simulate bayes optimization
        train_times = results[['unit', 'train__time']].set_index('unit').to_dict()['train__time']
        train_times = {tuple(k): train_times[k] for k in test_gen.units if k in train_times}

        assert len(train_times) > 0

        units, inputs, signals = zip(*[(u, v[:2, :], v) for u, v in test_gen.data.items()])

        aux = {u: (s, train_times[u]) for u, s in zip(units, signals) if u in train_times}
        test_datasets = set([u.split('_')[0] for u in test_gen.units])

        results = results[results.dataset.map(lambda x: x in test_datasets)]

        epochs_avoided = 0
        num_runs = 0
        num_prunings = 0
        total_epochs = 0
        total_train_time = 0
        avoided_train_time = 0
        experiments = results[['data__dataset_name', 'data__dataset_target', 'model__net']].drop_duplicates().values
        filter_best_losses = []
        real_best_losses = []

        if debug:
            experiments = experiments[:1]

        with tqdm.tqdm(total=len(units)) as progress_bar:
            for dataset, task, net in experiments:
                unit_ordered = list(
                    results[(results.dataset == dataset) & (results.net == net) & (results.task == task)].unit)

                if debug:
                    unit_ordered = unit_ordered[:10]

                filter_best_loss = 100000
                real_best_loss = 10000

                for unit in unit_ordered:

                    if unit in test_gen.units:
                        unit_data = aux[tuple(unit)]
                        curve = unit_data[0][:, 1]
                        real_val_loss = curve[-1]
                        real_epochs = unit_data[0].shape[0]

                        train_time = unit_data[1]
                        total_train_time += train_time
                        total_epochs += real_epochs
                        num_runs += 1

                        if real_val_loss < real_best_loss:  # real best loss found during real training
                            real_best_loss = real_val_loss

                        if random.uniform(0, 1) < random_pct:  # no stopped
                            if real_val_loss < filter_best_loss:  # best loss found during simulated training
                                filter_best_loss = real_val_loss

                        else:
                            epochs_avoided += real_epochs
                            avoided_train_time += ((train_time / real_epochs) * (real_epochs))
                            num_prunings += 1

                    progress_bar.update(1)

                filter_best_losses.append(filter_best_loss)
                real_best_losses.append(real_best_loss)

        csv_config["epochs_avoided"] = epochs_avoided
        csv_config["total_epochs"] = total_epochs
        csv_config["total_train_time"] = total_train_time
        csv_config["avoided_train_time"] = avoided_train_time
        csv_config["filter_best_losses"] = filter_best_losses
        csv_config["real_best_losses"] = real_best_losses
        csv_config["num_prunings"] = num_prunings
        csv_config["num_runs"] = num_runs

        csv_config["random_pct"] = random_pct
        csv_config["test_mse"] = 0

        logging.info(f"epochs_avoided: {epochs_avoided}")
        logging.info(f"total_train_time: {total_train_time}")
        logging.info(f"avoided_train_time: {avoided_train_time}")
        logging.info(f"filter_best_losses: {filter_best_losses}")
        logging.info(f"real_best_losses: {real_best_losses}")
        logging.info(
            f"fount best: {np.mean([np.abs(f - r) < 1e-06 for f, r in zip(filter_best_losses, real_best_losses)])}")
        logging.info(f"num_prunings: {num_prunings}")
        logging.info(f"num_runs: {num_runs}")

        csv_config["train__status"] = "FINISHED"

        history = {"test_mse": [0]}
        queue.put((history, arch_hash))

        log_train(csv_config, directory)

        logging.info("Finished train")

    except Exception as ex:
        if 'OOM' in str(ex):
            csv_config["train__status"] = "OOM ERROR"
        else:
            csv_config["train__status"] = "ERROR: " + str(ex)

        logging.error("Error: %s" % ex)
        logging.error(traceback.format_exc())
        sys.stdout.flush()
        queue.put(None)

        log_train(csv_config, directory)
