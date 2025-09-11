import os
import time
from typing import Callable
import numpy as np

import pickle as pk

import phmd
from ray import train as rtrain
import multiprocessing
import sys
import logging
import traceback
import pandas as pd
from filelock import FileLock
import phm_framework
from phm_framework.logging import secure_decode
from phm_framework.utils import flat_dict

logging.basicConfig(level=logging.INFO)


def log_train(config, directory):
    config = flat_dict(config)

    lock_file = os.path.join(directory, f'train.lock')
    log_file = os.path.join(directory, f'train.csv')
    with FileLock(lock_file) as lock:
        try:
            if os.path.exists(log_file):
                log = pd.read_csv(log_file)
                log = pd.concat([log, pd.DataFrame(data=[config])], ignore_index=True)
            else:
                log = pd.DataFrame(data=[config])

            logging.info("Saving log train csv")
            log.to_csv(log_file, index=False)
        finally:
            lock.release()


def load_log(net_name, directory):
    log_file = os.path.join(directory, f'train.csv')

    if os.path.exists(log_file):
        return pd.read_csv(log_file)
    else:
        return False


def get_best_info(net_name, data_name, monitor, directory):
    L = load_log(net_name, directory)
    L = L[(L.model__net == net_name) & (L.data__dataset_name == data_name)]

    best_hash = L.groupby('arch_hash')[monitor].mean().idxmin()
    best_score = L[L.arch_hash == best_hash][monitor].mean()
    best_std = L[L.arch_hash == best_hash][monitor].std()

    return best_hash, best_score, best_std




def parameter_opt_cv(model_creator: Callable,
                     experiment_config: dict = {},
                     trainer: Callable = None,
                     debug: bool = False):
    '''
        Configuración y ejecución de un experimento de optimización de parámetros utilizando validación cruzada
        Entrada:
            - experiment_config: diccionario que contiene la configuración del experimento
            - trainer: algoritmo de entrenamiento
            - debug: booleano que indica si se selecciona modo depuración
    '''
    try:
        training_config = experiment_config['train']
        output_dir = experiment_config['log']['directory']
        model_name = experiment_config['model']['net']
        data_name = experiment_config['data']['dataset_name']
        target = experiment_config['data']['dataset_target']

        if trainer is None:
            net_module = getattr(getattr(phm_framework, 'models'), model_name)
            trainer_class = getattr(net_module, 'TRAINER')
            trainer = trainer_class().train

        output_dir = os.path.join(output_dir, data_name, target, model_name)

        ds = phmd.datasets.Dataset(data_name)
        task = ds[target]

        # min_score = config.pop('min_score')
        stop_criteria = secure_decode(training_config, "stop_criteria", str, default=True, task=task.meta, pop=True)
        monitor = secure_decode(training_config, "monitor", str, default='val_loss', task=task.meta, pop=False)
        timeout = secure_decode(training_config, "timeout", int, default=None, task=task.meta, pop=False)
        num_folds = secure_decode(training_config, 'num_folds', int, default=5, task=task.meta, pop=False)

        experiment_config['train'] = training_config

        # wd = model_config.pop('working_dir')
        # os.chdir(wd)

        data = experiment_config.copy()
        data['model'] = data['model']['net'] if model_creator is None else model_creator.__name__
        data['folds'] = {}

        # cross-validation
        finish = False
        for ifold in range(num_folds):
            queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=trainer, args=(model_creator, experiment_config, ifold,
                                                              queue, debug, output_dir, timeout))

            p.start()
            p.join()
            if p.is_alive():
                logging.info('Fold %d timeout' % ifold)
                p.terminate()
                p.join()

                finish = True
            else:
                r = queue.get()
                if r is None:
                    finish = True

                else:
                    data['folds'][ifold] = r[0]
                    arch_hash = r[1]

            if len(data['folds'].keys()) > 0:
                # compute the mean score
                scores = [data['folds'][ifold][monitor] for ifold in data['folds'].keys()]
                scores = np.array(scores).flatten()

                rtrain.report({"score": np.mean(scores), "std_score": np.std(scores)})

            elif finish:
                logging.info("Not finished any trial")
                rtrain.report({"score": 999, "mean_epochs": 999, "std_score": 999, "nasa_score": 999, "mae": 999})

            if finish:
                logging.info("Finished train")
                return


        rtrain.report({"score": np.mean(scores), "std_score": np.std(scores)})

    except Exception as ex:
        logging.error("Error: %s" % ex)
        logging.error(traceback.format_exc())
        sys.stdout.flush()
        queue.put(None)
