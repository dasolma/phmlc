import os
import random
import time
from typing import Callable
import numpy as np
import phmd
import multiprocessing
import sys
import logging
import traceback
import pandas as pd
from filelock import FileLock
from phmd import datasets
from sklearn import model_selection, tree
import phm_framework
from phm_framework.logging import secure_decode
from phm_framework.models.utils import simplify_tree_recursive

from phm_framework.utils import flat_dict
import pickle as pk
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


            elif finish:
                logging.info("Not finished any trial")

            if finish:
                logging.info("Finished train")
                return

    except Exception as ex:
        logging.error("Error: %s" % ex)
        logging.error(traceback.format_exc())
        sys.stdout.flush()
        queue.put(None)


def parameter_opt_cv_v2(model_creator: Callable,
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
        random_state = training_config["random_state"]

        # extract test datasets
        ds = phmd.datasets.Dataset(data_name)
        task = ds['final_loss']  # Retrieve task-specific details
        task.random_state = random_state
        task.filters = {'data': 'curves'}

        X = task.load()[0]
        dataset_names = X.dataset.unique()
        random.shuffle(dataset_names)
        train_end_index = int(len(dataset_names) * 0.4)
        test_dataset_names = dataset_names[train_end_index:]

        del X
        experiment_config['data']['test_dataset_names'] = test_dataset_names

        if trainer is None:
            net_module = getattr(getattr(phm_framework, 'models'), model_name)
            trainer_class = getattr(net_module, 'TRAINER')
            trainer = trainer_class().train

        output_dir = os.path.join(output_dir, data_name, target, model_name)

        ds = phmd.datasets.Dataset(data_name)
        task = ds[target]

        timeout = secure_decode(training_config, "timeout", int, default=None, task=task.meta, pop=False)

        experiment_config['train'] = training_config

        data = experiment_config.copy()
        data['model'] = data['model']['net'] if model_creator is None else model_creator.__name__
        data['folds'] = {}

        # ACUMULADORES PARA POOLING GLOBAL
        hashes = []
        datas = []

        # Train neural network curve estimator
        # todo: debug
        num_folds = 1 if debug else 3
        csv_config = None
        LOCK_FILE = os.path.join(output_dir, 'net.lock')
        for ifold in range(num_folds):

            with FileLock(LOCK_FILE) as lock:
                try:
                    queue = multiprocessing.Queue()
                    if model_creator is not None:
                        args = (model_creator, experiment_config, ifold, queue, debug, output_dir, timeout)
                    else:
                        args = (experiment_config, ifold, queue, debug, output_dir, timeout)
                    p = multiprocessing.Process(target=trainer, args=args)

                    p.start()

                    # Primero vaciamos la queue ANTES de join()
                    try:
                        r = queue.get(timeout=timeout)  # Timeout como seguridad extra
                    except multiprocessing.queues.Empty:
                        r = None

                    p.join()
                    if p.is_alive():
                        logging.info('Finished train because fold %d timeout' % ifold)
                        p.terminate()
                        p.join()

                        return
                    else:
                        #r = queue.get()
                        if r is None:
                            logging.info("Finished train")
                            return

                        else:
                            data['folds'][ifold] = r[0]
                            arch_hash = r[1]
                            hashes.append(arch_hash)
                            datas.append(r[2])
                            csv_config = r[3]

                            logging.info(f"Finished training fold {ifold} of {arch_hash}")

                except Exception as ex:
                    logging.error("Error: %s" % ex)
                    logging.error(traceback.format_exc())
                    sys.stdout.flush()
                finally:
                    lock.release()
                    time.sleep(10)

        # Obtenemos las curvas
        Xs = [pk.load(open(datas[i], 'rb'))[0] for i in range(len(datas))]

        # Agregate discretized data generated by each network
        X = pd.concat(Xs).groupby(['unit', 'epoch']).mean().reset_index()
        X = X[~X.T.isnull().any()]

        # Split for train and validation (simulation)
        datasets = X.unit.map(lambda x: x[:[c.islower() for c in x].index(True)]).unique()
        # todo: /3
        ndatasets = len(datasets) // 3

        train_datasets = datasets[:ndatasets]
        val_datasets = datasets[ndatasets:2*ndatasets]

        X_train = X[X.unit.map(lambda x: x[:[c.islower() for c in x].index(True)] in train_datasets)]
        Y_train = (X_train['continue'] > 0.5).astype('bool')

        del X_train['continue']
        del X_train['unit']
        X_val = X[X.unit.map(lambda x: x[:[c.islower() for c in x].index(True)] in val_datasets)]
        del X_val['continue']

        # get optimization history
        ds = phmd.datasets.Dataset(data_name)
        _task = ds['final_loss']
        _task.filters = {"data": "results"}
        (opt_history,) = _task.load()

        from phm_framework.optimization.curves.train import find_optimal_strategy_tree, simulate_strategy, save_tree

        # curves
        curves = pk.load(open(datas[0], 'rb'))[1]
        optimal_tree, tree_params, val_sim_score = \
            find_optimal_strategy_tree(X_train, Y_train, X_val, curves, opt_history, output_dir)
        csv_config['val_sim_score'] = val_sim_score

        test_datasets = datasets[2*ndatasets:]
        X_test = X[X.unit.map(lambda x: x[:[c.islower() for c in x].index(True)] in test_datasets)]
        del X_test['continue']

        csv_config = simulate_strategy(X_test, curves, opt_history, optimal_tree, csv_config)

        log_train(csv_config, output_dir)
        save_tree(arch_hash, optimal_tree, output_dir, tree_params)

    except Exception as ex:
        logging.error("Error: %s" % ex)
        logging.error(traceback.format_exc())
        sys.stdout.flush()
        queue.put(None)



def parameter_opt_cv_fsldt(model_creator: Callable,
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
        random_state = training_config["random_state"]

        # extract test datasets
        ds = phmd.datasets.Dataset(data_name)
        task = ds['final_loss']  # Retrieve task-specific details
        task.random_state = random_state
        task.filters = {'data': 'curves'}

        X = task.load()[0]
        dataset_names = X.dataset.unique()
        random.shuffle(dataset_names)
        train_end_index = int(len(dataset_names) * 0.4)
        test_dataset_names = dataset_names[train_end_index:]

        del X
        experiment_config['data']['test_dataset_names'] = test_dataset_names

        if trainer is None:
            net_module = getattr(getattr(phm_framework, 'models'), model_name)
            trainer_class = getattr(net_module, 'TRAINER')
            trainer = trainer_class().train

        output_dir = os.path.join(output_dir, data_name, target, model_name)

        ds = phmd.datasets.Dataset(data_name)
        task = ds[target]

        timeout = secure_decode(training_config, "timeout", int, default=None, task=task.meta, pop=False)

        experiment_config['train'] = training_config

        data = experiment_config.copy()
        data['model'] = data['model']['net'] if model_creator is None else model_creator.__name__
        data['folds'] = {}

        # ACUMULADORES PARA POOLING GLOBAL
        hashes = []
        datas = []

        # Train neural network curve estimator
        # todo: debug
        num_folds = 1 if debug else 3
        csv_config = None
        LOCK_FILE = os.path.join(output_dir, 'net.lock')
        for ifold in range(num_folds):

            with FileLock(LOCK_FILE) as lock:
                try:
                    queue = multiprocessing.Queue()
                    if model_creator is not None:
                        args = (model_creator, experiment_config, ifold, queue, debug, output_dir, timeout)
                    else:
                        args = (experiment_config, ifold, queue, debug, output_dir, timeout)
                    p = multiprocessing.Process(target=trainer, args=args)

                    p.start()

                    # Primero vaciamos la queue ANTES de join()
                    try:
                        r = queue.get(timeout=timeout)  # Timeout como seguridad extra
                    except multiprocessing.queues.Empty:
                        r = None

                    p.join()
                    if p.is_alive():
                        logging.info('Finished train because fold %d timeout' % ifold)
                        p.terminate()
                        p.join()

                        return
                    else:
                        #r = queue.get()
                        if r is None:
                            logging.info("Finished train")
                            return

                        else:
                            data['folds'][ifold] = r[0]
                            arch_hash = r[1]
                            hashes.append(arch_hash)
                            datas.append(r[2])
                            csv_config = r[3]

                            logging.info(f"Finished training fold {ifold} of {arch_hash}")

                except Exception as ex:
                    logging.error("Error: %s" % ex)
                    logging.error(traceback.format_exc())
                    sys.stdout.flush()
                finally:
                    lock.release()
                    time.sleep(10)

        csv_config['pn_data'] = data['folds']

        # Obtenemos las curvas
        Xs = [pk.load(open(datas[i], 'rb'))[0] for i in range(len(datas))]

        # Agregate discretized data generated by each network
        X = pd.concat(Xs).groupby(['unit', 'epoch']).mean().reset_index()
        X = X[~X.T.isnull().any()]

        # Split for train and validation (simulation)
        datasets = X.unit.map(lambda x: x[:[c.islower() for c in x].index(True)]).unique()
        ndatasets = len(datasets) // 3

        train_datasets = datasets[:ndatasets]
        val_datasets = datasets[ndatasets:2*ndatasets]

        X_train = X[X.unit.map(lambda x: x[:[c.islower() for c in x].index(True)] in train_datasets)]
        Y_train = (X_train['continue'] > 0.5).astype('bool')

        del X_train['continue']
        del X_train['unit']
        X_val = X[X.unit.map(lambda x: x[:[c.islower() for c in x].index(True)] in val_datasets)]
        del X_val['continue']

        # get optimization history
        ds = phmd.datasets.Dataset(data_name)
        _task = ds['final_loss']
        _task.filters = {"data": "results"}
        (opt_history,) = _task.load()

        from phm_framework.optimization.curves.train import save_tree
        from phm_framework.optimization.curves.fsldt import simulate_strategy_optuna, find_optimal_strategy_tree

        # curves
        curves = pk.load(open(datas[0], 'rb'))[1]
        optimal_tree, tree_params, val_sim_score, val_best_rank, val_rank_pct, epochs_saved_pct = \
            find_optimal_strategy_tree(X_train, Y_train, X_val, curves, opt_history, output_dir, debug)
        csv_config['mean_train_val_score'] = val_sim_score
        csv_config['mean_train_rank'] = val_best_rank
        csv_config['train_rank_pct'] = val_rank_pct
        csv_config['train_epochs_saved_pct'] = epochs_saved_pct

        test_datasets = datasets[2*ndatasets:]
        X_test = X[X.unit.map(lambda x: x[:[c.islower() for c in x].index(True)] in test_datasets)]
        del X_test['continue']


        test_sim_score, _, _, test_best_rank, test_rank_pct, epochs_saved_pct = \
            simulate_strategy_optuna(X_test, curves, opt_history, optimal_tree, csv_config)
        csv_config['mean_test_val_score'] = test_sim_score
        csv_config['mean_test_rank'] = test_best_rank
        csv_config['test_rank_pct'] = test_rank_pct
        csv_config['test_epochs_saved_pct'] = epochs_saved_pct

        log_train(csv_config, output_dir)
        save_tree(arch_hash, optimal_tree, output_dir, tree_params)

    except Exception as ex:
        logging.error("Error: %s" % ex)
        logging.error(traceback.format_exc())
        sys.stdout.flush()
        queue.put(None)


def parameter_opt_cv_fsldt(model_creator: Callable,
                           experiment_config: dict = {},
                           trainer: Callable = None,
                           debug: bool = False):
    '''
        Configuración y ejecución de un experimento de optimización de parámetros utilizando validación cruzada
    '''
    try:
        training_config = experiment_config['train']
        output_dir = experiment_config['log']['directory']
        model_name = experiment_config['model']['net']
        data_name = experiment_config['data']['dataset_name']
        target = experiment_config['data']['dataset_target']
        random_state = training_config["random_state"]

        # Extract test datasets
        ds = phmd.datasets.Dataset(data_name)
        task = ds['final_loss']
        task.random_state = random_state
        task.filters = {'data': 'curves'}

        X = task.load()[0]
        dataset_names = X.dataset.unique()
        random.shuffle(dataset_names)
        train_end_index = int(len(dataset_names) * 0.4)
        test_dataset_names = dataset_names[train_end_index:]

        del X
        experiment_config['data']['test_dataset_names'] = test_dataset_names

        if trainer is None:
            net_module = getattr(getattr(phm_framework, 'models'), model_name)
            trainer_class = getattr(net_module, 'TRAINER')
            trainer = trainer_class().train

        output_dir = os.path.join(output_dir, data_name, target, model_name)

        ds = phmd.datasets.Dataset(data_name)
        task = ds[target]

        timeout = secure_decode(training_config, "timeout", int, default=None, task=task.meta, pop=False)
        experiment_config['train'] = training_config

        data = experiment_config.copy()
        data['model'] = data['model']['net'] if model_creator is None else model_creator.__name__
        data['folds'] = {}

        hashes = []
        datas = []
        num_folds = 1 if debug else 3
        csv_config = None
        LOCK_FILE = os.path.join(output_dir, 'net.lock')

        # ═══════════════ NUEVA SECCIÓN PARALELA ═══════════════
        processes = []
        queues = []

        # Envolvemos todo el grupo de folds en el LOCK para que corran juntos de forma exclusiva
        with FileLock(LOCK_FILE) as lock:
            try:
                # 1. Lanzar TODOS los procesos en paralelo (Non-blocking)
                for ifold in range(num_folds):
                    queue = multiprocessing.Queue()
                    if model_creator is not None:
                        args = (model_creator, experiment_config, ifold, queue, debug, output_dir, timeout)
                    else:
                        args = (experiment_config, ifold, queue, debug, output_dir, timeout)

                    p = multiprocessing.Process(target=trainer, args=args)
                    p.start()

                    processes.append((ifold, p))
                    queues.append((ifold, queue))

                logging.info(f"Lanzados {num_folds} folds simultáneamente en la CPU.")

                # 2. Recolectar resultados conforme vayan terminando
                raw_results = {}
                for ifold, queue in queues:
                    try:
                        # Al estar ya todos corriendo, el .get() esperará al proceso correspondiente
                        r = queue.get(timeout=timeout)
                    except multiprocessing.queues.Empty:
                        r = None
                    raw_results[ifold] = r

                # 3. Control de vida y cierre seguro de los procesos
                for ifold, p in processes:
                    p.join(timeout=5)
                    if p.is_alive():
                        logging.warning(f'Fold {ifold} superó el timeout de ejecución. Forzando terminación.')
                        p.terminate()
                        p.join()

                # 4. Procesar y estructurar los datos recuperados
                for ifold in range(num_folds):
                    r = raw_results[ifold]
                    if r is None:
                        logging.error(f"El Fold {ifold} falló o no devolvió datos. Abortando experimento.")
                        return

                    data['folds'][ifold] = r[0]
                    arch_hash = r[1]
                    hashes.append(arch_hash)
                    datas.append(r[2])
                    csv_config = r[3]  # Se preserva el último csv_config de la iteración tal como en tu código

                    logging.info(f"Procesado con éxito el Fold {ifold} de {arch_hash}")

            except Exception as ex:
                logging.error("Error en la ejecución paralela: %s" % ex)
                logging.error(traceback.format_exc())
                sys.stdout.flush()
                return
            finally:
                lock.release()
                time.sleep(5)  # Un pequeño respiro tras liberar los recursos

            # ═══════════════ FIN DE LA SECCIÓN PARALELA ═══════════════

            csv_config['pn_data'] = data['folds']

            # Obtenemos las curvas
            Xs = [pk.load(open(datas[i], 'rb'))[0] for i in range(len(datas))]

            # Agregate discretized data generated by each network
            X = pd.concat(Xs).groupby(['unit', 'epoch']).mean().reset_index()
            X = X[~X.T.isnull().any()]

            # Split for train and validation (simulation)
            datasets = X.unit.map(lambda x: x[:[c.islower() for c in x].index(True)]).unique()
            ndatasets = len(datasets) // 3

            train_datasets = datasets[:ndatasets]
            val_datasets = datasets[ndatasets:2 * ndatasets]

            X_train = X[X.unit.map(lambda x: x[:[c.islower() for c in x].index(True)] in train_datasets)]
            Y_train = (X_train['continue'] > 0.5).astype('bool')

            del X_train['continue']
            del X_train['unit']
            X_val = X[X.unit.map(lambda x: x[:[c.islower() for c in x].index(True)] in val_datasets)]
            del X_val['continue']

            # Get optimization history
            ds = phmd.datasets.Dataset(data_name)
            _task = ds['final_loss']
            _task.filters = {"data": "results"}
            (opt_history,) = _task.load()

            from phm_framework.optimization.curves.train import save_tree
            from phm_framework.optimization.curves.fsldt import simulate_strategy_optuna, find_optimal_strategy_tree

            # Curves
            curves = pk.load(open(datas[0], 'rb'))[1]
            optimal_tree, tree_params, val_sim_score, val_best_rank, val_rank_pct, epochs_saved_pct = \
                find_optimal_strategy_tree(X_train, Y_train, X_val, curves, opt_history, output_dir, debug)
            csv_config['mean_train_val_score'] = val_sim_score
            csv_config['mean_train_rank'] = val_sim_score
            csv_config['train_rank_pct'] = val_rank_pct
            csv_config['train_epochs_saved_pct'] = epochs_saved_pct

            test_datasets = datasets[2 * ndatasets:]
            X_test = X[X.unit.map(lambda x: x[:[c.islower() for c in x].index(True)] in test_datasets)]
            del X_test['continue']

            test_sim_score, _, _, test_best_rank, test_rank_pct, epochs_saved_pct = \
                simulate_strategy_optuna(X_test, curves, opt_history, optimal_tree, csv_config)
            csv_config['mean_test_val_score'] = test_sim_score
            csv_config['mean_test_rank'] = test_best_rank
            csv_config['test_rank_pct'] = test_rank_pct
            csv_config['test_epochs_saved_pct'] = epochs_saved_pct

            log_train(csv_config, output_dir)
            save_tree(arch_hash, optimal_tree, output_dir, tree_params)

    except Exception as ex:
        logging.error("Error global en parameter_opt_cv_fsldt: %s" % ex)
        logging.error(traceback.format_exc())
        sys.stdout.flush()
def parameter_opt_cv_hb(model_creator: Callable,
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
        random_state = training_config["random_state"]

        # extract test datasets
        ds = phmd.datasets.Dataset(data_name)
        task = ds['final_loss']  # Retrieve task-specific details
        task.random_state = random_state
        task.filters = {'data': 'curves'}

        X = task.load()[0]
        dataset_names = X.dataset.unique()
        random.shuffle(dataset_names)
        train_end_index = int(len(dataset_names) * 0.4)
        test_dataset_names = dataset_names[train_end_index:]

        del X
        experiment_config['data']['test_dataset_names'] = test_dataset_names

        if trainer is None:
            net_module = getattr(getattr(phm_framework, 'models'), model_name)
            trainer_class = getattr(net_module, 'TRAINER')
            trainer = trainer_class().train

        output_dir = os.path.join(output_dir, data_name, target, model_name)

        ds = phmd.datasets.Dataset(data_name)
        task = ds[target]

        timeout = secure_decode(training_config, "timeout", int, default=None, task=task.meta, pop=False)
        num_folds = secure_decode(training_config, 'num_folds', int, default=5, task=task.meta, pop=False)

        experiment_config['train'] = training_config

        data = experiment_config.copy()
        data['model'] = data['model']['net'] if model_creator is None else model_creator.__name__
        data['folds'] = {}

        # ACUMULADORES PARA POOLING GLOBAL
        hashes = []
        datas = []

        # Train neural network curve estimator
        # todo: debug
        num_folds = 3
        csv_config = None
        LOCK_FILE = os.path.join(output_dir, 'net.lock')
        for ifold in range(num_folds):

            with FileLock(LOCK_FILE) as lock:
                try:
                    queue = multiprocessing.Queue()
                    if model_creator is not None:
                        args = (model_creator, experiment_config, ifold, queue, debug, output_dir, timeout)
                    else:
                        args = (experiment_config, ifold, queue, debug, output_dir, timeout)
                    p = multiprocessing.Process(target=trainer, args=args)

                    p.start()

                    # Primero vaciamos la queue ANTES de join()
                    try:
                        r = queue.get(timeout=timeout)  # Timeout como seguridad extra
                    except multiprocessing.queues.Empty:
                        r = None

                    p.join()
                    if p.is_alive():
                        logging.info('Finished train because fold %d timeout' % ifold)
                        p.terminate()
                        p.join()

                        return
                    else:
                        #r = queue.get()
                        if r is None:
                            logging.info("Finished train")
                            return

                        else:
                            logging.info(f"Finished training fold {ifold}")

                except Exception as ex:
                    logging.error("Error: %s" % ex)
                    logging.error(traceback.format_exc())
                    sys.stdout.flush()
                finally:
                    lock.release()
                    time.sleep(10)


    except Exception as ex:
        logging.error("Error: %s" % ex)
        logging.error(traceback.format_exc())
        sys.stdout.flush()
        queue.put(None)



def train_rule_tree(X, Y, params, negative_class_threshold):
    valid_cols = [c for c in X.columns if c != 'best_performance']
    clf = tree.DecisionTreeClassifier(max_depth=params['max_depth'],
                                      min_samples_leaf=params['min_samples'],
                                      class_weight={False: 1, True: params['continue_weight']})
    clf.fit(X[valid_cols], Y)
    if negative_class_threshold >= 0.5:
        simplify_tree_recursive(clf.tree_, negative_class_threshold=negative_class_threshold)
    p = clf.predict(X[valid_cols])
    facc, tacc = (p == Y).values[np.where(~Y)].mean(), (p == Y).values[np.where(Y)].mean()

    return clf, tacc, facc
