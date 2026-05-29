import optuna
import sys
import traceback
import warnings
import os
import pandas as pd
import numpy as np
from phm_framework.logging import confighash, HASH_EXCLUDE, load_log, secure_decode, log_train
from phm_framework.optimization.curves import load_curves
from phm_framework.trainers.utils import get_task
from phm_framework.utils import flat_dict
import logging
from joblib import Parallel, delayed


# Desactivar logs de Optuna para no ensuciar tu consola
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Simulador BOHB (Bayesian Optimization + Hyperband) sobre curvas históricas
# ══════════════════════════════════════════════════════════════════════════════

class BOHBSimulator:
    """
    Simula BOHB utilizando Optuna (TPESampler + HyperbandPruner).

    A diferencia de Hyperband puro, BOHB necesita saber los hiperparámetros
    asociados a cada 'unit_id' para que el modelo Bayesiano pueda aprender.
    """

    def __init__(
            self,
            val_curves: dict,
            unit_params: dict,  # NUEVO: {unit_id: {'lr': 0.01, 'batch_size': 32, ...}}
            R: int = None,
            eta: int = 3,
            minimize: bool = True,
    ):
        if not val_curves:
            raise ValueError("val_curves está vacío.")

        self.val_curves = val_curves
        self.unit_params = unit_params
        self.all_units = list(val_curves.keys())
        self.R = R or int(max(len(c) for c in val_curves.values()))
        self.eta = eta
        self.minimize = minimize

        # Para tabular benchmarks, pre-calculamos la mejor curva por configuración
        # para agilizar la búsqueda del vecino más cercano.
        self.param_df = pd.DataFrame.from_dict(unit_params, orient='index')

        self.__categorical_params = ['model__activation', 'model__batch_normalization', 'model__conv_activation',
                              'model__dense_activation']

        params = list(self.unit_params.values())[0].keys()
        self.params_ranges = {}
        for k in params:
            values = np.array([d[k] for d in self.unit_params.values()])
            if k in self.__categorical_params:
                self.params_ranges[k] = np.unique(values)
            else:
                self.params_ranges[k] = (values.min(), values.max())

        # ══════ OPTIMIZACIÓN NUMPY ══════
        self.param_df = pd.DataFrame.from_dict(unit_params, orient='index')
        # Guardamos las columnas numéricas para el cálculo de distancia
        self.numeric_cols = self.param_df.select_dtypes(include=[np.number]).columns.tolist()

        # Convertimos a matrices NumPy nativas (búsqueda ultra rápida)
        self.matrix_uids = self.param_df.index.to_numpy()
        self.matrix_values = self.param_df[self.numeric_cols].to_numpy()

    @classmethod
    def from_group(
            cls,
            group_df: pd.DataFrame,
            unit_params_dict: dict,  # El diccionario global {unit_id: {hp1: v1, ...}}
            R: int = None,
            eta: int = 3,
            minimize: bool = True,
    ) -> "BOHBSimulator":
        """
        Construye un simulador BOHB desde el subDataFrame de un grupo.
        Filtra automáticamente los parámetros para quedarse SOLO con las 'units' de este grupo.
        """
        # 1. Extraer las curvas de validación de este grupo específico
        val_curves = (
            group_df
            .sort_index()
            .groupby("unit", sort=False)["val_loss"]
            .apply(list)
            .to_dict()
        )

        # 2. FILTRADO CRUCIAL: Nos quedamos solo con los parámetros de las units de ESTE grupo
        group_units = set(val_curves.keys())
        filtered_params = {
            uid: unit_params_dict[uid]
            for uid in group_units
            if uid in unit_params_dict
        }

        # Control de errores por si alguna unidad no tiene sus hiperparámetros registrados
        missing = group_units - set(filtered_params.keys())
        if missing:
            filtered_params = {k: v for k, v in filtered_params.items() if k not in missing}
            group_units = list(filtered_params.keys())
            val_curves = {k: v for k, v in val_curves.items() if k in group_units}

        # 3. Inicializamos la clase pasándole únicamente el set de datos de este grupo
        return cls(val_curves, filtered_params, R=R, eta=eta, minimize=minimize)

    @classmethod
    def run_all_groups(
            cls,
            df: pd.DataFrame,
            unit_params_dict: dict,
            group_cols: list = None,
            R: int = None,
            eta: int = 3,
            minimize: bool = True,
            n_runs: int = 20,
            n_trials: int = 100,  # <-- El budget de trials para Optuna por cada run
            seed: int = 42,
            n_jobs: int = -1,
    ) -> pd.DataFrame:

        group_cols = group_cols or ["dataset", "task", "net"]
        grouped = list(df.groupby(group_cols, sort=True))

        # Definimos una función interna para procesar UN SOLO GRUPO de forma aislada
        def _process_single_group(keys, group_df):
            key_dict = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
            try:
                sim = cls.from_group(group_df, unit_params_dict, R=R, eta=eta, minimize=minimize)
                mc_results = sim.run_montecarlo(n_runs=n_runs, n_trials=n_trials, seed=seed)
                mc_results = sim.enrich_results(mc_results)

                for col, val in key_dict.items():
                    mc_results[col] = val
                return mc_results
            except Exception as exc:
                print(f"[BOHBSimulator] Grupo {key_dict} omitido: {exc}")
                return None

        # ══════ EJECUCIÓN EN PARALELO ══════
        # Distribuye los grupos automáticamente entre tus cores de la CPU
        results_list = Parallel(n_jobs=n_jobs)(
            delayed(_process_single_group)(keys, group_df) for keys, group_df in grouped
        )

        # Filtrar los omitidos (None) y concatenar
        records = [r for r in results_list if r is not None]

        if not records:
            raise RuntimeError("Ningún grupo pudo simularse.")

        col_order = group_cols + [
            "run", "best_unit", "best_val_loss",
            "rank", "rank_pct", "epochs_used",
            "epochs_saved", "val_score"
        ]
        result = pd.concat(records, ignore_index=True)
        return result[[c for c in col_order if c in result.columns]]

    def _unit_final_score(self, uid) -> float:
        return float(self.val_curves[uid][-1])

    def unit_rankings(self) -> dict:
        scores = {uid: self._unit_final_score(uid) for uid in self.all_units}
        sorted_units = sorted(scores, key=lambda u: scores[u], reverse=not self.minimize)
        return {uid: rank + 1 for rank, uid in enumerate(sorted_units)}

    def enrich_results(self, mc_df):
        n_units = len(self.all_units)
        rankings = self.unit_rankings()
        mc_df = mc_df.copy()
        mc_df["rank"] = mc_df["best_unit"].map(rankings)
        mc_df["rank_pct"] = (mc_df["rank"] - 1) / max(n_units - 1, 1)
        mc_df["epochs_saved"] = mc_df["baseline_epochs"] - mc_df["epochs_used"]
        return mc_df

    # ── lógica de optuna (BOHB) ───────────────────────────────────────────────

    def _find_closest_unit(self, suggested_params: dict) -> str:
        """
        Encuentra el unit_id pre-evaluado más cercano a lo que sugiere el TPE.
        Si tu espacio en Optuna es exactamente igual al grid de tus datos,
        esto encontrará una coincidencia exacta.
        """
        # 1. Intentar coincidencia exacta rápida via diccionario
        # (Si tu espacio de búsqueda coincide con tu grid, esto toma tiempo O(1))
        for uid, params in self.unit_params.items():
            if all(params.get(k) == v for k, v in suggested_params.items()):
                return uid

        # 2. Si no hay coincidencia exacta, distancia Euclídea vectorizada en NumPy
        suggested_vector = np.array([suggested_params[col] for col in self.numeric_cols])

        # Operación matricial broadcasting en C (mucho más rápido que Pandas)
        distances = np.sum((self.matrix_values - suggested_vector) ** 2, axis=1)
        return self.matrix_uids[np.argmin(distances)]

    def run_once(self, n_trials=50, seed=42):
        """
        Ejecuta un 'Study' de Optuna que simula BOHB.
        n_trials: número de configuraciones a evaluar (equivalente al budget de HB).
        """
        sampler = optuna.samplers.TPESampler(seed=seed)
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=self.R,
            reduction_factor=self.eta
        )

        direction = "minimize" if self.minimize else "maximize"
        study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)

        epochs_used = 0
        baseline_epochs = 0
        real_best_score = np.inf
        sampled_units_all = list()
        # ══════ OPTIMIZACIÓN DE RUNGS ══════
        # Pre-calcular las épocas exactas donde Hyperband toma decisiones: 1, eta, eta^2, eta^3...
        rungs = set()
        curr_rung = 1
        while curr_rung <= self.R:
            rungs.add(curr_rung)
            curr_rung *= self.eta

        def objective(trial):
            nonlocal epochs_used, baseline_epochs, real_best_score

            suggested = {}
            for k, v in self.params_ranges.items():
                if k in self.__categorical_params:
                    suggested[k] = trial.suggest_categorical(k, v)
                else:
                    suggested[k] = trial.suggest_float(k, v[0], v[1], log=v[0] != 0)

            uid = self._find_closest_unit(suggested)
            sampled_units_all.append(uid)
            curve = self.val_curves[uid]

            baseline_epochs += len(curve)

            last_val_loss = None
            max_steps = min(self.R, len(curve))

            for epoch_idx in range(max_steps):
                step = epoch_idx + 1
                val_loss = curve[epoch_idx]
                last_val_loss = val_loss
                epochs_used += 1

                # Reportamos la métrica a Optuna obligatoriamente
                trial.report(val_loss, step)

                # ══════ LLAMADA FILTRADA A LA PODA ══════
                # Solo preguntamos a Optuna si debe podar si estamos en un Rung o en la última época
                if step in rungs or step == max_steps:
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            if real_best_score > curve[-1]:
                real_best_score = curve[-1]

            return last_val_loss

        # Ejecutamos la simulación
        warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
        try:
            study.optimize(objective, n_trials=n_trials)
        except Exception as e:
            pass  # Captura errores por si todos los trials son podados

        # Extraer el mejor resultado
        best_trial = study.best_trial

        # Recuperamos el unit_id real de los parámetros del mejor trial
        best_uid = self._find_closest_unit(best_trial.params)
        best_score = best_trial.value

        baseline_epochs = sum(len(self.val_curves[u]) for u in sampled_units_all)

        rank = sorted([np.array(self.val_curves[u]).min() for u in sampled_units_all])
        rank = [r >= best_score for r in rank].index(True)
        print(f"Best uid {best_uid}, Best score: {best_score}, Epochs saved: {(baseline_epochs - epochs_used) / baseline_epochs:0.2f}, rank: {rank}")

        performance_score = real_best_score / best_score

        total_epochs = baseline_epochs + epochs_used
        time_score = (epochs_used / total_epochs)

        score = (0.5 * performance_score + 0.5 * time_score)

        return best_uid, best_score, epochs_used, baseline_epochs, score

    def run_montecarlo(self, n_runs=20, n_trials=50, seed=42):
        results = []
        for i in range(n_runs):
            uid, val_loss, ep, baseline, val_score = self.run_once(n_trials=n_trials, seed=seed + i)
            results.append({
                "run": i,
                "best_unit": uid,
                "best_val_loss": val_loss,
                "epochs_used": ep,
                "baseline_epochs": baseline,
                "val_score": val_score,
            })
        return pd.DataFrame(results)


def bohb_simulation(config, ifold, queue, debug, directory, timeout):

    try:
        training_config = config['train']
        data_config     = config['data']

        data_name   = data_config['dataset_name']
        data_target = data_config['dataset_target']
        task        = get_task(data_name, data_target, None)

        csv_config = flat_dict(config.copy())
        csv_config['train__max_epochs'] = csv_config.pop('train__epochs')
        csv_config['train__fold']       = ifold
        nhash     = confighash(csv_config, exclude=HASH_EXCLUDE)
        arch_hash = confighash(csv_config, exclude=HASH_EXCLUDE + ["train__fold"])
        csv_config['run_hash']  = nhash
        csv_config['arch_hash'] = arch_hash

        if not os.path.exists(directory):
            os.makedirs(directory)

        log_csv = load_log(None, directory)
        if not isinstance(log_csv, bool):
            query = log_csv[log_csv.run_hash == nhash]
            if query.shape[0] > 0 and query.iloc[0].train__status == 'FINISHED':
                r = query.iloc[0]
                queue.put(({'val_loss': [r.val_loss], 'test_loss': [r.test_loss]}, arch_hash))
                return

        logging.info("Reading data")
        random_state = secure_decode(training_config, "random_state", dtype=int, task=task)

        sets = load_curves(ifold,
                           num_folds=config['train']['num_folds'],
                           normalize_output=False,
                           filters={"data": "curves"},
                           test_dataset_names=config['data']['test_dataset_names'],
                           random_state=random_state)

        fold_train = sets['train']
        fold_val = sets['val']
        train_df = pd.concat((fold_train, fold_val))
        test_df  = sets['test']


        hp_set  = load_curves(None,
                           num_folds=0,
                           normalize_output=False,
                           filters={"data": "results"},
                           test_dataset_names=None,
                           random_state=random_state)

        # 1. Identificas las columnas que representan tus hiperparámetros (ej: lr, batch_size, dropout...)
        # Si tus columnas de hiperparámetros no tienen prefijo, puedes listarlas explícitamente:
        unit_params_dict = create_unit_params_dict(fold_train.unit, hp_set)

        group_cols  = ["dataset", "task", "net"]
        eta_grid    = training_config.get('hyperband_eta_grid', [2, 3])
        n_runs      = training_config.get('hyperband_n_runs',   30)
        hb_seed     = training_config.get('hyperband_seed',     42)

        # ── paso 1: CV sobre train para seleccionar (R, eta) ─────────────────
        logging.info("Hyperband CV: optimizing R and eta")

        best = None
        R = 100
        for eta in eta_grid:

            results = BOHBSimulator.run_all_groups(
                fold_train,
                unit_params_dict=unit_params_dict,
                group_cols=group_cols,
                R=R, eta=eta,
                n_runs=n_runs,
                seed=hb_seed,
            )
            mean_rp = results["val_score"].mean()
            if best is None or mean_rp < best["rank_pct"]:
                best = {"R": R, "eta": eta, "rank_pct": mean_rp}

                print(best)

        best_R   = best['R']
        best_eta = best['eta']
        logging.info(f"Best params → R={best_R}, eta={best_eta} "
                     f"(mean_rank_pct={best['rank_pct']:.3f})")

        # ── paso 2: Hyperband sobre train completo con los mejores params ─────
        logging.info("Hyperband: running on full train set")

        unit_params_dict = create_unit_params_dict(train_df.unit, hp_set)
        train_results = BOHBSimulator.run_all_groups(
            train_df,
            unit_params_dict=unit_params_dict,
            group_cols=group_cols,
            R=best_R, eta=best_eta, n_runs=n_runs, seed=hb_seed,
        )

        # ── paso 3: evaluación del ganador en test ────────────────────────────
        logging.info("Hyperband: running on test set with best params")

        unit_params_dict = create_unit_params_dict(test_df.unit, hp_set)
        test_results = BOHBSimulator.run_all_groups(
            test_df,
            unit_params_dict=unit_params_dict,
            group_cols=group_cols,
            R=best_R, eta=best_eta, n_runs=n_runs, seed=hb_seed,
        )

        # ── métricas globales y persistencia ─────────────────────────────────────

        mean_test_saved_pct = (test_results["epochs_saved"] / (
                    test_results["epochs_saved"] + test_results["epochs_used"])).mean()
        mean_test_val_loss = test_results["best_val_loss"].mean()
        mean_test_rank_pct = test_results["rank_pct"].mean()
        mean_test_val_score = test_results["val_score"].mean()

        test_mean_rank = test_results['rank'].mean()
        mean_train_saved_pct = (train_results["epochs_saved"] / (train_results["epochs_saved"] + train_results["epochs_used"])).mean()
        mean_train_rank_pct = train_results["rank_pct"].mean()
        mean_train_val_score = train_results["val_score"].mean()

        mean_epochs_saved = train_results["epochs_saved"].mean()
        train_mean_rank = train_results['rank'].mean()

        prefix = os.path.join(directory, nhash)
        train_results.to_csv(f"{prefix}_train.csv", index=False)
        test_results.to_csv(f"{prefix}_test.csv", index=False)

        csv_config.update({
            "train__status": "FINISHED",
            "fold": ifold,
            "best_R": best_R,
            "best_eta": best_eta,
            "test_rank_pct": mean_test_rank_pct,
            "train_rank_pct": mean_train_rank_pct,
            "epochs_saved": mean_epochs_saved,
            "test_epochs_saved_pct": mean_test_saved_pct,
            "train_epochs_saved_pct": mean_train_saved_pct,
            "mean_test_val_score": mean_test_val_score,
            "mean_train_val_score": mean_train_val_score,
            "test_mean_rank": test_mean_rank,
            "train_mean_rank": train_mean_rank,
            "random_state": random_state

        })
        log_train(csv_config, directory)

        queue.put(({'val_loss': [mean_test_val_loss],
                    'test_loss': [mean_test_rank_pct]}, arch_hash))

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


def create_unit_params_dict(units, X):
    X = X[X.unit.isin(units)]

    hp_cols = [c for c in X.columns if 'model__' in c]
    # 2. Creas el diccionario global mapeando cada 'unit' con sus valores de hiperparámetros
    unit_params_dict = X.groupby('unit')[hp_cols].first().to_dict('index')
    unit_params_dict = {u: {hp: v for hp, v in hps.items() if str(v) != 'nan' and v is not None}
                        for u, hps in unit_params_dict.items()}
    unit_params_dict = {u: {hp: v for hp, v in hps.items() if
                            hp not in ['model__input_shape', 'model__output', 'model__net', 'model__output_dim']}
                        for u, hps in unit_params_dict.items()}

    return unit_params_dict
