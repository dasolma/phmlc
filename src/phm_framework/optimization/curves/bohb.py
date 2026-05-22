import optuna
import sys
import traceback
import warnings
import os
import pandas as pd
from phm_framework.logging import confighash, HASH_EXCLUDE, load_log, secure_decode, log_train
from phm_framework.optimization.curves import load_curves
from phm_framework.trainers.utils import get_task
from phm_framework.utils import flat_dict
import logging


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

    @classmethod
    def from_group(
            cls,
            group_df: pd.DataFrame,
            unit_params_dict: dict,
            R: int = None,
            eta: int = 3,
            minimize: bool = True,
    ) -> "BOHBSimulator":
        """Construye el simulador BOHB desde un subDataFrame."""
        val_curves = (
            group_df
            .sort_index()
            .groupby("unit", sort=False)["val_loss"]
            .apply(list)
            .to_dict()
        )
        return cls(val_curves, unit_params_dict, R=R, eta=eta, minimize=minimize)

    # ── ranking (Igual que en tu código) ──────────────────────────────────────

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
        # Búsqueda exacta rápida
        for uid, params in self.unit_params.items():
            if all(params.get(k) == v for k, v in suggested_params.items()):
                return uid

        # Si el TPE sugiere valores continuos, calculamos la distancia euclídea
        # (Asegúrate de normalizar si tienes escalas muy distintas en tu dataset real)
        diffs = self.param_df - pd.Series(suggested_params)
        distances = (diffs ** 2).sum(axis=1)
        return distances.idxmin()

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
        sampled_units_all = set()

        def objective(trial):
            nonlocal epochs_used

            # 1. BOHB sugiere hiperparámetros
            # ¡IMPORTANTE!: Ajusta este bloque según los verdaderos hiperparámetros de tus datos
            suggested = {
                'lr': trial.suggest_float('lr', 1e-5, 1e-1, log=True),
                # 'batch_size': trial.suggest_categorical('batch_size',),
                # Añade aquí los mismos que definen tus 'unit_params'
            }

            # 2. Buscamos la curva histórica correspondiente
            uid = self._find_closest_unit(suggested)
            sampled_units_all.add(uid)
            curve = self.val_curves[uid]

            # 3. Simulamos el entrenamiento época a época para que el Pruner actúe
            last_val_loss = None
            for epoch_idx in range(min(self.R, len(curve))):
                step = epoch_idx + 1
                val_loss = curve[epoch_idx]
                last_val_loss = val_loss

                epochs_used += 1
                trial.report(val_loss, step)

                # Successive Halving en acción: corta el entrenamiento si es malo
                if trial.should_prune():
                    raise optuna.TrialPruned()

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

        return best_uid, best_score, epochs_used, baseline_epochs

    def run_montecarlo(self, n_runs=20, n_trials=50, seed=42):
        results = []
        for i in range(n_runs):
            uid, score, ep, baseline = self.run_once(n_trials=n_trials, seed=seed + i)
            results.append({
                "run": i,
                "best_unit": uid,
                "best_val_loss": score,
                "epochs_used": ep,
                "baseline_epochs": baseline,
            })
        return pd.DataFrame(results)


def bohb_simulation(config, ifold, queue, debug, directory, timeout):

    try:
        training_config = config['train']
        random_state = training_config['random_state']
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

        group_cols  = ["dataset", "task", "net"]
        R_grid      = training_config.get('hyperband_R_grid',   [9, 27, 81])
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
                group_cols=group_cols,
                R=R, eta=eta,
                n_runs=n_runs,
                seed=hb_seed,
            )
            mean_rp = results["rank_pct"].mean()
            if best is None or mean_rp < best["rank_pct"]:
                best = {"R": R, "eta": eta, "rank_pct": mean_rp}

                print(best)

        best_R   = best['R']
        best_eta = best['eta']
        logging.info(f"Best params → R={best_R}, eta={best_eta} "
                     f"(mean_rank_pct={best['rank_pct']:.3f})")

        # ── paso 2: Hyperband sobre train completo con los mejores params ─────
        logging.info("Hyperband: running on full train set")

        train_results = BOHBSimulator.run_all_groups(
            train_df, group_cols=group_cols,
            R=best_R, eta=best_eta, n_runs=n_runs, seed=hb_seed,
        )

        # ── paso 3: evaluación del ganador en test ────────────────────────────
        logging.info("Hyperband: running on test set with best params")

        test_results = BOHBSimulator.run_all_groups(
            test_df, group_cols=group_cols,
            R=best_R, eta=best_eta, n_runs=n_runs, seed=hb_seed,
        )

        # ── métricas globales y persistencia ─────────────────────────────────────

        mean_test_saved_pct = (test_results["epochs_saved"] / (
                    test_results["epochs_saved"] + test_results["epochs_used"])).mean()
        mean_test_val_loss = test_results["best_val_loss"].mean()
        mean_test_rank_pct = test_results["rank_pct"].mean()
        test_mean_rank = test_results['rank'].mean()
        mean_train_saved_pct = (train_results["epochs_saved"] / (train_results["epochs_saved"] + train_results["epochs_used"])).mean()
        mean_train_rank_pct = train_results["rank_pct"].mean()
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

