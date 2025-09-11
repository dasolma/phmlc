import logging
import os
from hashlib import sha1

import pandas as pd
from filelock import FileLock

from phm_framework import typing
from phm_framework.utils import flat_dict

logging.basicConfig(level=logging.INFO)


HASH_EXCLUDE = ["run_hash", "arch_hash", "log__directory", "log__save_only_best", "train__verbose"]


def secure_decode(config, key, dtype, task, default=None, pop=True):
    if key not in config:
        return default

    if pop:
        value = config.pop(key)
    else:
        value = config[key]

    value = typing.ensure_param(value, dtype, task)

    return value
def confighash(config, exclude=[]):
    if exclude is not None and len(exclude) > 0:
        config = config.copy()
        for key in exclude:
            if key in config:
                del config[key]

    return sha1(repr(sorted(config.items())).encode()).hexdigest()


def log_train(config, directory):
    config = flat_dict(config)

    lock_file = os.path.join(directory, f'train.lock')
    log_file = os.path.join(directory, f'train.csv')
    with FileLock(lock_file) as lock:
        try:
            if os.path.exists(log_file):
                try:
                    log = pd.read_csv(log_file)
                except:
                    log = pd.DataFrame(data=[config])
                log = pd.concat([log, pd.DataFrame(data=[config])], ignore_index=True)
            else:
                log = pd.DataFrame(data=[config])

            logging.info("Saving log train csv")
            log.to_csv(log_file, index=False)
        finally:
            lock.release()

def get_results(net_hash, directory):
    log_file = os.path.join(directory, f'train.csv')

    if os.path.exists(log_file):
        try:
            log = pd.read_csv(log_file)
        except:
            return False

        mask = (log.run_hash == net_hash) & (log.train__status == 'FINISHED')
        if mask.any():
            result_cols = [c for c in log.columns if 'val_' in c or 'test_' in c]
            r = log.loc[mask, result_cols].iloc[0].to_dict()
            r = {k: [v] for k, v in r.items()}
            return r

    return False


def load_log(net_name, directory):
    log_file = os.path.join(directory, f'train.csv')

    return pd.read_csv(log_file)


def get_rows(log, **kwargs):
    if log is None:
        return 0

    log['model__net'] = log.model__net.map(lambda x: x.replace("['", "").replace("']", ""))

    mask = None
    for k, v in kwargs.items():
        if mask is None:
            mask = log[k] == v
        else:
            mask &= log[k] == v

    mask &= (log.train__status == 'FINISHED')

    _m = log[mask]
    return _m.shape[0]

def get_best_info(net_name, data_name, monitor, directory):
    L = load_log(net_name, directory)
    L = L[(L.model__net == net_name) & (L.data__dataset_name == data_name)]

    best_hash = L.groupby('arch_hash')[monitor].mean().idxmin()
    best_score = L[L.arch_hash == best_hash][monitor].mean()
    best_std = L[L.arch_hash == best_hash][monitor].std()

    return best_hash, best_score, best_std
