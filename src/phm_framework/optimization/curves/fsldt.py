import copy
import importlib
import itertools
import multiprocessing
import os
import sys
import traceback
import pickle as pk
from phmd import datasets
from phm_framework.data import FSLSequence
from phm_framework.data.generators import load_train_net_generators_v2
from phm_framework.logging import confighash, HASH_EXCLUDE, secure_decode, log_train
from phm_framework.optimization.curves.train import extended_decision_data
from phm_framework.optimization.utils import train_rule_tree
from phm_framework.trainers.utils import get_task
from phm_framework.utils import flat_dict
from sklearn.tree import _tree, export_text
import numpy as np
import pandas as pd
import optuna
import logging
import warnings
from typing import Dict, List, Tuple, Optional

# Variables globales temporales para que cada núcleo de la CPU acceda a los datos sin copiarlos
_w_cls = None
_w_Xtest = None
_w_curves = None
_w_opt_history = None


def _init_experiment_worker(cls, Xtest, curves, opt_history):
    """Inicializa la memoria compartida de cada proceso hijo."""
    global _w_cls, _w_Xtest, _w_curves, _w_opt_history
    _w_cls = cls
    _w_Xtest = Xtest
    _w_curves = curves
    _w_opt_history = opt_history


def _run_parallel_experiment(args):
    """Ejecuta de forma aislada la simulación de un único experimento."""
    experiment_id, seed, minimize = args

    # Construir el simulador usando la referencia global de la clase y los datos
    sim = _w_cls.from_experiment(experiment_id, _w_Xtest, _w_opt_history, _w_curves, minimize=minimize)
    if sim is None:
        return None

    # Ejecutar simulación BO + Pruning
    f_best, r_best, rank_idx, l_metrics = sim.run_once(seed=seed)

    # Extraer los datos de tiempos históricos aquí mismo para aprovechar la CPU en paralelo
    eresults = _w_opt_history[_w_opt_history.unit.str.contains(experiment_id, na=False)]
    t_time = eresults[~eresults.train__time.isnull()].train__time.sum()

    return {
        "experiment_id": experiment_id,
        "f_best": f_best,
        "r_best": r_best,
        "rank_idx": rank_idx,
        "l_metrics": l_metrics,
        "t_time": t_time,
        "unit_ordered_len": min(100, len(sim.unit_ordered))
    }

class BOPredictiveSimulator:
    def __init__(
            self,
            exp_hp_df: pd.DataFrame,
            Xexp: pd.DataFrame,
            curves_dict: dict,
            minimize: bool = True
    ):
        """
        Inicializa el simulador para UN experimento/grupo específico.
        """
        self.Xexp = Xexp
        self.curves_dict = curves_dict
        self.minimize = minimize
        self.unit_ordered = list(exp_hp_df.index)

        # Guardar mapeo de hiperparámetros completo para el control de rangos
        self.exp_hp_df = exp_hp_df

        hp_cols = [c for c in exp_hp_df.columns if 'model__' in c]
        hp_cols = [c for c in hp_cols if c not in ['model__input_shape', 'model__output', 'model__net',
                                                   'model__output_dim']]
        exclude_cols = []
        for c in hp_cols:
            if exp_hp_df[c].isnull().all():
                exclude_cols.append(c)
            elif 'activation' in c:
                exp_hp_df[c] = exp_hp_df[c].map(lambda x: 'leakyReLU' if 'LeakyReLU' in x else x)
        hp_cols = [c for c in hp_cols if c not in exclude_cols]
        self.hp_cols = hp_cols

        # ══════ OPTIMIZACIÓN NUMPY (Búsqueda Vectorizada) ══════
        self.numeric_cols = exp_hp_df[hp_cols].select_dtypes(include=[np.number]).columns.tolist()
        self.matrix_uids = exp_hp_df.index.to_numpy()
        self.matrix_values = exp_hp_df[self.numeric_cols].to_numpy()

        # Extraer el espacio de búsqueda (rangos y categorías) dinámico para Optuna
        self.categorical_params = ['model__activation', 'model__batch_normalization', 'model__conv_activation',
                                   'model__dense_activation', 'model__kernel_size', 'model__bidirectional',
                                   'model__cell_type']
        self.param_ranges = {}
        #self.categorical_params = []
        for col in self.hp_cols:
            if col in self.categorical_params:
                self.param_ranges[col] = exp_hp_df[col].unique().tolist()
            else:
                self.param_ranges[col] = (float(exp_hp_df[col].min()), float(exp_hp_df[col].max()))

    @classmethod
    def from_experiment(
            cls,
            experiment_id: str,
            Xtest_global: pd.DataFrame,
            opt_history_global: pd.DataFrame,
            curves_list: List,
            minimize: bool = True
    ) -> Optional["BOPredictiveSimulator"]:
        """
        Método de factoría (Factory Method) equivalente a 'from_group'.
        Aísla y filtra los datos globales correspondientes a un único experimento.
        """
        # Filtrar historial de optimización
        eresults = opt_history_global[opt_history_global.unit.str.contains(experiment_id, na=False)].copy()
        unit_ordered = list(eresults.unit)

        # Filtrar dataset de pruebas (Xtest)
        Xexp = Xtest_global[Xtest_global.unit.isin(unit_ordered)].copy()
        unit_ordered = [u for u in unit_ordered if u in Xexp.unit.values]

        eresults = eresults[eresults.unit.isin(unit_ordered)]
        eresults = eresults[~eresults.train__time.isnull()]

        if len(unit_ordered) <= 1:
            return None  # No hay suficientes unidades para optimizar secuencialmente

        # Indexar las curvas de este grupo específico en un diccionario O(1)
        curves_dict = {d[0]: d[-2] for d in curves_list if d[0] in unit_ordered}


        # Preparar dataframe indexado por unidad para el entrenamiento
        exp_hp_df = eresults.set_index('unit')

        # Enriquecer Xexp con la metadata de épocas finales (equivalente a tu merge original)
        real_epochs = pd.DataFrame([
            {'unit': uid, 'epochs': curve.shape[0], 'final_val_loss': curve[-1][1]}
            for uid, curve in curves_dict.items()
        ])
        Xexp = pd.merge(Xexp, real_epochs, on='unit')

        return cls(exp_hp_df, Xexp, curves_dict, minimize=minimize)

    def _find_closest_unit(self, suggested_params: dict) -> str:
        """Encuentra el identificador de la unidad más cercana usando NumPy."""
        suggested_vector = np.array([suggested_params[col] for col in self.numeric_cols])
        distances = np.sum((self.matrix_values - suggested_vector) ** 2, axis=1)
        return self.matrix_uids[np.argmin(distances)]

    def run_once(self, seed: int = 42) -> Tuple[float, Dict]:
        """
        Ejecuta la simulación de optimización bayesiana guiada por el clasificador
        para este experimento concreto. Equivalente a 'run_once' de BOHB.
        """
        # Inicializadores de métricas locales
        metrics = {
            "epochs_avoided": 0,
            "total_epochs": 0,
            "total_train_time": 0,
            "saved_time": 0,
            "total_time": 0,
            "num_prunings": 0,
            "num_runs": 0
        }

        best_loss = np.inf
        filter_best_loss = np.inf
        worst_loss = 0
        all_losses = []


        # Configurar Optuna de forma aislada
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize" if self.minimize else "maximize", sampler=sampler)

        def objective(trial):
            nonlocal filter_best_loss, best_loss, worst_loss, all_losses

            # 1. Sugerir hiperparámetros
            suggested = {}
            for k, v in self.param_ranges.items():
                if k in self.categorical_params:
                    suggested[k] = trial.suggest_categorical(k, v)
                else:
                    suggested[k] = trial.suggest_float(k, v[0], v[1], log=v[0] != 0)

            # 2. Encontrar vecino más cercano
            unit = self._find_closest_unit(suggested)

            curve_data = self.curves_dict[unit]

            # Best loss
            all_losses.append(curve_data[-1][1])

            if best_loss > curve_data[-1][1]:
                best_loss = curve_data[-1][1]

            if worst_loss < curve_data[-1][1]:
                worst_loss = curve_data[-1][1]

            # 3. Evaluar simulación de parada con el clasificador
            decision_data = self.Xexp[self.Xexp.unit == unit].sort_values('epoch')
            preds = decision_data.pred.values
            ureal_epochs = len(self.curves_dict[unit]) # TODO: revisar len(preds)


            patience = 3
            stop_counter = 0
            final_stop_epoch = ureal_epochs
            is_pruned = False

            for i, p in enumerate(preds):
                if not p:
                    stop_counter += 1
                else:
                    stop_counter = 0

                if stop_counter >= patience:
                    final_stop_epoch = i + 1
                    is_pruned = True
                    break

            epochs_to_run = final_stop_epoch if is_pruned else ureal_epochs
            uepochs_avoided = ureal_epochs - epochs_to_run

            # Acumular métricas
            metrics["total_epochs"] += ureal_epochs
            metrics["epochs_avoided"] += uepochs_avoided


            # El tiempo de entrenamiento
            epoch_time = self.exp_hp_df.loc[unit].train__time
            if hasattr(epoch_time, "__len__"):  # Si es un array/serie debido a duplicados
                epoch_time = epoch_time.mean()

            metrics['saved_time'] += epoch_time * uepochs_avoided
            metrics['total_time'] += epoch_time * ureal_epochs

            # Recuperamos el train__time real mapeando al DataFrame original que guardamos
            # Nota: para simplificar la lectura, asumimos que se inyecta o se lee directamente
            metrics["num_runs"] += 1
            if is_pruned:
                metrics["num_prunings"] += 1


            # Si está podado, reportamos hasta final_stop_epoch. Si no, reportamos toda la curva real.
            steps_to_report = final_stop_epoch if is_pruned else ureal_epochs

            for step_idx in range(steps_to_report):
                # El paso en Optuna suele empezar en 1 (o en 0, sé consistente con tu diseño)
                trial.report(curve_data[step_idx][1], step=step_idx + 1)

            if is_pruned:
                raise optuna.TrialPruned()
            else:
                final_loss = curve_data[-1][1]
                if final_loss < filter_best_loss:
                    filter_best_loss = final_loss
                return final_loss

        # Lanzar optimización para los ensayos restantes (N - 1)
        n_trials = min(100, len(self.unit_ordered))
        if n_trials > 0:
            study.optimize(objective, n_trials=n_trials)

        real_best_loss = best_loss #self.Xexp.best_performance.min()

        if filter_best_loss == np.inf:  # se podaron todas!
            logging.info("All trials pruned!")
            filter_best_loss = worst_loss

        rank_loss_idx = sorted(all_losses).index(filter_best_loss)

        return filter_best_loss, real_best_loss, rank_loss_idx, metrics

    def run_once(self, seed: int = 42) -> Tuple[float, float, int, Dict]:
        """
        Ejecuta la simulación reproduciendo los trials en el mismo orden cronológico
        en el que ocurrieron en la ejecución real mediante un bucle nativo.
        """
        # Inicializadores de métricas locales
        metrics = {
            "epochs_avoided": 0,
            "total_epochs": 0,
            "total_train_time": 0,
            "saved_time": 0,
            "total_time": 0,
            "num_prunings": 0,
            "num_runs": 0
        }

        best_loss = np.inf
        filter_best_loss = np.inf
        worst_loss = 0
        all_losses = []

        # Determinamos cuántos trials vamos a reproducir (máximo 100)
        n_trials = min(100, len(self.unit_ordered))

        # 1. El bucle nativo que reemplaza a study.optimize()
        for idx in range(n_trials):
            unit = self.unit_ordered[idx]
            curve_data = self.curves_dict[unit]

            # Guardar pérdidas para el histórico
            final_trial_loss = curve_data[-1][1]
            all_losses.append(final_trial_loss)

            if best_loss > final_trial_loss:
                best_loss = final_trial_loss

            if worst_loss < final_trial_loss:
                worst_loss = final_trial_loss

            # 2. Evaluar simulación de parada con el clasificador
            decision_data = self.Xexp[self.Xexp.unit == unit].sort_values('epoch')
            preds = decision_data.pred.values
            ureal_epochs = len(curve_data)

            patience = 3
            stop_counter = 0
            final_stop_epoch = ureal_epochs
            is_pruned = False

            for i, p in enumerate(preds):
                if not p:
                    stop_counter += 1
                else:
                    stop_counter = 0

                if stop_counter >= patience:
                    final_stop_epoch = i + 1
                    is_pruned = True
                    break

            epochs_to_run = final_stop_epoch if is_pruned else ureal_epochs
            uepochs_avoided = ureal_epochs - epochs_to_run

            # Acumular métricas
            metrics["total_epochs"] += ureal_epochs
            metrics["epochs_avoided"] += uepochs_avoided

            # Obtener el tiempo de entrenamiento
            epoch_time = self.exp_hp_df.loc[unit].train__time
            if hasattr(epoch_time, "__len__") and not isinstance(epoch_time, str):
                epoch_time = epoch_time.mean()

            metrics['saved_time'] += epoch_time * uepochs_avoided
            metrics['total_time'] += epoch_time * ureal_epochs

            metrics["num_runs"] += 1
            if is_pruned:
                metrics["num_prunings"] += 1

            # 3. Lógica del filtro de pérdida
            if is_pruned:
                # Si está podado, su pérdida se congela en el momento de la parada
                # Nota: Asegúrate de indexar correctamente según tu estructura (ej: final_stop_epoch - 1)
                pruned_loss = curve_data[final_stop_epoch - 1][1]
                # La pérdida podada no compite por ser la mejor real, pero sí afecta al progreso
            else:
                # Si completó el entrenamiento, compite por ser el mejor filter_best_loss
                if final_trial_loss < filter_best_loss:
                    filter_best_loss = final_trial_loss

        # 4. Control de seguridad por si se podaron absolutamente todos los trials
        real_best_loss = best_loss

        if filter_best_loss == np.inf:
            logging.info("All trials pruned!")
            filter_best_loss = worst_loss

        rank_loss_idx = sorted(all_losses).index(filter_best_loss)

        return filter_best_loss, real_best_loss, rank_loss_idx, metrics

    @classmethod
    def run_all_experiments(
            cls,
            Xtest: pd.DataFrame,
            curves: list,
            opt_history: pd.DataFrame,
            clf,
            minimize: bool = True,
            seed: int = 42,
            csv_config: dict = None
    ):
        """
        Orquesta la simulación distribuyendo los experimentos en paralelo a lo largo de la CPU.
        """
        warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        Xtest = Xtest.copy()
        Xtest['pred'] = clf.predict(Xtest[clf.feature_names_in_])

        # Identificar los experimentos válidos
        experiments_in_test = Xtest.unit.map(lambda x: "_".join(x.split("_")[:-1])).unique()
        experiments_in_curves = set(['_'.join(e[0].split('_')[:-1]) for e in curves])
        experiments = [e for e in experiments_in_test if e in experiments_in_curves]

        # Acumuladores globales
        g_metrics = {
            "epochs_avoided": 0, "total_epochs": 0, "total_train_time": 0,
            "avoided_train_time": 0, "num_prunings": 0, "num_runs": 0
        }
        filter_best_losses = []
        real_best_losses = []
        rank_losses = []
        rank_pct = []
        epochs_saved_pct = []
        saved_time = []
        total_time = []
        valid_experiments = []

        # ═══════════════ PROCESAMIENTO PARALELO ═══════════════
        num_workers = max(1, multiprocessing.cpu_count() - 1)

        # Preparamos las tareas empaquetadas
        tasks = [(exp_id, seed, minimize) for exp_id in experiments]

        # Lanzamos el Pool inyectando las matrices pesadas una sola vez por Core
        pool = multiprocessing.Pool(
            processes=num_workers,
            initializer=_init_experiment_worker,
            initargs=(cls, Xtest, curves, opt_history)
        )

        try:
            raw_results = pool.map(_run_parallel_experiment, tasks)
        finally:
            pool.close()
            pool.join()
        # ═══════════════════════════════════════════════════════

        # Consolidar y acumular los resultados (Bucle secuencial ultra-rápido de agregación)
        for res in raw_results:
            if res is None:
                continue

            experiment_id = res["experiment_id"]
            f_best = res["f_best"]
            r_best = res["r_best"]
            rank_idx = res["rank_idx"]
            l_metrics = res["l_metrics"]
            t_time = res["t_time"]

            valid_experiments.append(experiment_id)
            filter_best_losses.append(f_best)
            real_best_losses.append(r_best)
            rank_losses.append(rank_idx)

            rank_pct.append(rank_idx / res["unit_ordered_len"])
            epochs_saved_pct.append(l_metrics['epochs_avoided'] / l_metrics['total_epochs'])
            saved_time.append(l_metrics['saved_time'])
            total_time.append(l_metrics['total_time'])

            # Consolidar acumuladores numéricos del diccionario global
            for k in g_metrics.keys():
                if k == "total_train_time":
                    g_metrics[k] += t_time
                elif k == "avoided_train_time":
                    ratio = l_metrics["epochs_avoided"] / max(1, l_metrics["total_epochs"])
                    g_metrics[k] += (t_time * ratio)
                else:
                    g_metrics[k] += l_metrics[k]

        # Calcular Scores finales
        performance_score = np.mean([r / f for r, f in zip(real_best_losses, filter_best_losses)])
        total_epochs = max(1, g_metrics["total_epochs"])
        time_score = g_metrics["epochs_avoided"] / total_epochs
        #score = (0.5 * performance_score + 0.5 * time_score)
        score = np.sqrt(performance_score * time_score)

        if csv_config is not None:
            for k, v in g_metrics.items():
                csv_config[k] = v
            csv_config["experiments"] = valid_experiments
            csv_config["filter_best_losses"] = filter_best_losses
            csv_config["real_best_losses"] = real_best_losses
            csv_config["rank_losses"] = rank_losses
            csv_config["saved_time"] = saved_time
            csv_config["total_time"] = total_time
            csv_config["train__status"] = "FINISHED"
            csv_config["score"] = score

            logging.info(f"epochs_avoided: {g_metrics['epochs_avoided']}")
            logging.info(
                f"found best: {np.mean([(np.abs(f - r) < 1e-06) or (f < r) for f, r in zip(filter_best_losses, real_best_losses)])}")

        return score, performance_score, time_score, np.mean(rank_losses), np.mean(rank_pct), \
               np.mean(epochs_saved_pct)

def curves_fsldt(model_creator, config, ifold, queue, debug, directory, timeout):
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


def simulate_strategy_optuna(Xtest, curves, opt_history, clf, csv_config=None):
    Xtest = Xtest.copy()

    resultado_experimento = BOPredictiveSimulator.run_all_experiments(
        Xtest=Xtest,
        curves=curves,
        opt_history=opt_history,
        clf=clf,
        seed=42,
        csv_config=csv_config  # Opcional, si pasas None te devuelve los scores puros
    )

    return resultado_experimento


def find_optimal_strategy_tree(X_train, Y_train, X_val, curves, opt_history, directory, debug=False):
    """
    Busca el árbol que maximiza el éxito de la estrategia global.
    """
    best_score = -np.inf
    best_tree = None
    best_params = None
    best_rank = None
    best_rank_pct = None

    print(f"ACC(T)\tACC(F)\tScore\tPerf.Score\tT.Score")
    print(f"------\t------\t-----\t----------\t-------")

    rules_founds = []
    params = itertools.product([0.45, 0.6, 0.75, 0.9],   #np.arange(0.45, 1.0, 0.05),
                                [2, 3, 4],
                                [50, 100, 200],
                                [10, 30, 60, 90] #range(10, 100, 10)
                               )
    if debug:
        params = [(0.45,2,50,10)]
    for negative_class_threshold, max_depth, min_samples, w in params:

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
        current_strategy_score, performance_score, time_score, mean_rank, mean_rank_pct, epochs_saved_pct = \
            simulate_strategy_optuna(X_val, curves, opt_history, candidate_tree)

        print(
            f" {tacc:0.2f}\t {facc:0.2f}\t{current_strategy_score:0.3f}\t{performance_score:0.3f}\t{time_score:0.3f}", end="")
        # 3. Guardar el mejor
        if current_strategy_score > best_score:
            best_score = current_strategy_score
            best_tree = candidate_tree
            best_params = copy.deepcopy(params)
            best_rank = mean_rank
            best_rank_pct = mean_rank_pct
            print("*")
        else:
            print("")


    return best_tree, best_params, best_score, best_rank, best_rank_pct, epochs_saved_pct

