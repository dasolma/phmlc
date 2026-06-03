import copy
import importlib
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

            "num_prunings": 0,
            "num_runs": 0
        }

        # Condición inicial: Unidad base del histórico (Unidad 0)
        init_unit = self.unit_ordered[0]
        filter_best_loss = self.Xexp[self.Xexp.unit == init_unit].final_val_loss.iloc[0]

        # Configurar Optuna de forma aislada
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize" if self.minimize else "maximize", sampler=sampler)

        def objective(trial):
            nonlocal filter_best_loss

            # 1. Sugerir hiperparámetros
            suggested = {}
            for k, v in self.param_ranges.items():
                if k in self.categorical_params:
                    suggested[k] = trial.suggest_categorical(k, v)
                else:
                    suggested[k] = trial.suggest_float(k, v[0], v[1], log=v[0] != 0)

            # 2. Encontrar vecino más cercano
            unit = self._find_closest_unit(suggested)

            # 3. Evaluar simulación de parada con el clasificador
            decision_data = self.Xexp[self.Xexp.unit == unit].sort_values('epoch')
            preds = decision_data.pred.values
            ureal_epochs = len(preds)

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


            # Recuperamos el train__time real mapeando al DataFrame original que guardamos
            # Nota: para simplificar la lectura, asumimos que se inyecta o se lee directamente
            metrics["num_runs"] += 1
            if is_pruned:
                metrics["num_prunings"] += 1

            curve_data = self.curves_dict[unit]

            if is_pruned:
                for step_idx in range(final_stop_epoch):
                    trial.report(curve_data[step_idx][1], step=step_idx + 1)
                raise optuna.TrialPruned()
            else:
                final_loss = curve_data[-1][1]
                if final_loss < filter_best_loss:
                    filter_best_loss = final_loss
                return final_loss

        # Lanzar optimización para los ensayos restantes (N - 1)
        n_trials = len(self.unit_ordered) - 1
        if n_trials > 0:
            study.optimize(objective, n_trials=n_trials)

        real_best_loss = self.Xexp.best_performance.min()
        rank_loss_idx = sorted(self.Xexp.groupby('unit').final_val_loss.max().values).index(filter_best_loss)

        return filter_best_loss, real_best_loss, rank_loss_idx, metrics

    @classmethod
    def run_all_experiments(
            cls,
            Xtest: pd.DataFrame,
            curves: List,
            opt_history: pd.DataFrame,
            clf,
            minimize: bool = True,
            seed: int = 42,
            csv_config: Optional[dict] = None
    ):
        """
        Orquesta la simulación iterando sobre todos los experimentos disponibles.
        """
        warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        Xtest = Xtest.copy()
        Xtest['pred'] = clf.predict(Xtest[clf.feature_names_in_])

        # Identificar los experimentos válidos que cruzan entre curvas e historial
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

        for experiment_id in experiments:
            # Construir el simulador aislado del experimento usando la factoría
            sim = cls.from_experiment(experiment_id, Xtest, opt_history, curves, minimize)
            if sim is None:
                continue

            # Ejecutar simulación BO + Pruning
            f_best, r_best, rank_idx, l_metrics = sim.run_once(seed=seed)

            # Consolidar colecciones
            filter_best_losses.append(f_best)
            real_best_losses.append(r_best)
            rank_losses.append(rank_idx)

            # Consolidar acumuladores numéricos dinámicamente
            for k in g_metrics.keys():
                # Nota: Para el train_time, extraemos el valor real de opt_history del experimento actual
                if k == "total_train_time" or k == "avoided_train_time":
                    # Cálculo proporcional del tiempo según épocas evitadas del experimento
                    eresults = opt_history[opt_history.unit.str.contains(experiment_id, na=False)]
                    t_time = eresults[~eresults.train__time.isnull()].train__time.sum()
                    if k == "total_train_time":
                        g_metrics[k] += t_time
                    else:
                        # Estimación del tiempo evitado relativo al ratio de épocas del grupo
                        ratio = l_metrics["epochs_avoided"] / max(1, l_metrics["total_epochs"])
                        g_metrics[k] += (t_time * ratio)
                else:
                    g_metrics[k] += l_metrics[k]

        # Calcular Scores finales de la estrategia comparada
        performance_score = np.mean([r / f for r, f in zip(real_best_losses, filter_best_losses)])

        total_epochs = max(1, g_metrics["total_epochs"])
        epochs_used = total_epochs - g_metrics["epochs_avoided"]
        time_score = epochs_used / total_epochs
        score = (0.5 * performance_score + 0.5 * time_score)

        if csv_config is not None:
            # Rellenar diccionario de configuración/salida
            for k, v in g_metrics.items():
                csv_config[k] = v
            csv_config["experiments"] = experiments
            csv_config["filter_best_losses"] = filter_best_losses
            csv_config["real_best_losses"] = real_best_losses
            csv_config["rank_losses"] = rank_losses
            csv_config["train__status"] = "FINISHED"
            csv_config["score"] = score

            # Logging informativo idéntico a tu función original
            logging.info(f"epochs_avoided: {g_metrics['epochs_avoided']}")
            logging.info(
                f"found best: {np.mean([(np.abs(f - r) < 1e-06) or (f < r) for f, r in zip(filter_best_losses, real_best_losses)])}")
            return csv_config

        return score, performance_score, time_score

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


def simulate_strategy_optuna(Xtest, curves, opt_history, clf, hp_cols=None, csv_config=None):
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
                    current_strategy_score, performance_score, time_score = simulate_strategy_optuna(X_val, curves, opt_history, candidate_tree)

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

