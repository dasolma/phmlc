import sys
import traceback
import warnings
import os
from phm_framework.logging import confighash, HASH_EXCLUDE, load_log, secure_decode, log_train
from phm_framework.optimization.curves import load_curves
from phm_framework.trainers.utils import get_task
from phm_framework.utils import flat_dict
import logging

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Simulador de Hyperband sobre curvas históricas
# ══════════════════════════════════════════════════════════════════════════════
import math
import random
import pandas as pd
from collections import defaultdict
import numpy as np

class HyperbandSimulator:
    """
    Simula Hyperband sin entrenar ninguna red.

    Lee las curvas de validación ya almacenadas y simula qué
    configuraciones habría descartado/mantenido Hyperband en
    cada ronda de Successive Halving.

    Uso típico
    ──────────
        df = pd.read_csv("curves.csv")
        results = HyperbandSimulator.run_all_groups(df, n_runs=30)

        # Resumen por grupo
        results.groupby(["dataset", "task", "net"]).agg(
            mean_rank      =("rank",         "mean"),
            pct_top10      =("rank_pct",     lambda x: (x <= 0.1).mean()),
            mean_ep_saved  =("epochs_saved", "mean"),
        )
    """

    # ── construcción ──────────────────────────────────────────────────────────

    def __init__(
        self,
        val_curves: dict,
        unit_times_dict: dict,
        R: int = None,
        eta: int = 3,
        minimize: bool = True,
    ):
        """
        Args:
            val_curves : {unit_id: [val_loss_epoch1, val_loss_epoch2, ...]}
            R          : épocas máximas (None → máximo entre todas las curvas)
            eta        : factor de reducción (≥ 2, típicamente 3)
            minimize   : True → menor val_loss es mejor
        """
        if not val_curves:
            raise ValueError("val_curves está vacío; el grupo no tiene datos.")

        self.val_curves = val_curves
        self.unit_times = unit_times_dict
        self.best_score = np.array([c[-1] for c in val_curves.values()]).min()
        self.all_units  = list(val_curves.keys())
        self.R          = R or int(np.mean([len(c) for c in val_curves.values()]))
        self.eta        = eta
        self.minimize   = minimize
        self.s_max      = math.floor(math.log(self.R, self.eta))

    @classmethod
    def from_group(
        cls,
        group_df: pd.DataFrame,
        unit_times_dict: dict,
        R: int = None,
        eta: int = 3,
        minimize: bool = True,
    ) -> "HyperbandSimulator":
        """
        Construye un simulador desde el subDataFrame de un grupo
        (dataset, task, net).

        Las filas se ordenan por (unit, orden de aparición) para preservar
        la secuencia de épocas sin necesitar una columna de época explícita.
        """
        val_curves = (
            group_df
            .sort_index()
            .groupby("unit", sort=False)["val_loss"]
            .apply(list)
            .to_dict()
        )
        return cls(val_curves, R=R, eta=eta, minimize=minimize, unit_times_dict=unit_times_dict)

    @classmethod
    def run_all_groups(
        cls,
        df: pd.DataFrame,
        unit_times_dict: dict,
        group_cols: list = None,
        R: int = None,
        eta: int = 3,
        minimize: bool = True,
        n_runs: int = 20,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Ejecuta la simulación Monte Carlo para cada (dataset, task, net)
        y devuelve los resultados enriquecidos con rank y epochs_saved.

        Returns
        ───────
        DataFrame con columnas:
            dataset, task, net,
            run, best_unit, best_val_loss, epochs_used,
            rank, rank_pct, epochs_saved
        """
        group_cols = group_cols or ["dataset", "task", "net"]
        records    = []

        for keys, group_df in df.groupby(group_cols, sort=True):
            key_dict = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))

            try:
                sim        = cls.from_group(group_df, R=R, eta=eta, minimize=minimize, unit_times_dict=unit_times_dict)
                mc_results = sim.run_montecarlo(n_runs=n_runs, seed=seed)
                mc_results = sim.enrich_results(mc_results)   # ← rank + epochs_saved

                for col, val in key_dict.items():
                    mc_results[col] = val

                records.append(mc_results)

            except Exception as exc:
                print(f"[HyperbandSimulator] Grupo {key_dict} omitido: {exc}")

        if not records:
            raise RuntimeError("Ningún grupo pudo simularse.")

        col_order = group_cols + [
            "run", "best_unit", "best_val_loss",
            "rank", "rank_pct", "epochs_used",
            "epochs_saved", "val_score", "filter_val_loss",
            "saved_time", "total_time"
        ]
        result = pd.concat(records, ignore_index=True)
        result['experiment'] = result.dataset + "_" + result.task + "_" + result.net
        return result[[c for c in col_order if c in result.columns]]

    # ── ranking y coste ───────────────────────────────────────────────────────

    def _unit_final_score(self, uid) -> float:
        """
        Val_loss en la última época disponible.
        Representa el rendimiento real del modelo completamente entrenado.
        Si hubo early stopping, es el score en el punto de parada.
        """
        return float(self.val_curves[uid][-1])

    @property
    def full_training_epochs(self) -> int:
        """
        Épocas totales si se hubieran entrenado todas las configuraciones
        hasta su duración máxima. Es el baseline de coste contra el que
        se mide el ahorro de Hyperband.
        """
        return sum(len(c) for c in self.val_curves.values())

    def unit_rankings(self) -> dict:
        """
        Ranking de cada unit por su val_loss final (1 = mejor configuración).

        El ranking se basa en la última época de cada curva, que refleja
        el rendimiento del modelo completo (con o sin early stopping real).

        Returns
        ───────
        {unit_id: rank}   rank ∈ [1, n_units]
        """
        scores       = {uid: self._unit_final_score(uid) for uid in self.all_units}
        sorted_units = sorted(scores, key=lambda u: scores[u], reverse=not self.minimize)
        return {uid: rank + 1 for rank, uid in enumerate(sorted_units)}

    def enrich_results(self, mc_df):
        mc_df = mc_df.copy()
        mc_df["rank_pct"] = mc_df["rank"] / 100
        mc_df["epochs_saved"] = mc_df["baseline_epochs"] - mc_df["epochs_used"]

        return mc_df

    # ── helpers internos ──────────────────────────────────────────────────────

    def _score_at(self, uid, target_epoch: int) -> float:
        curve = self.val_curves[uid]
        idx   = min(target_epoch - 1, len(curve) - 1)
        return float(curve[idx])

    def _actual_epochs_added(self, uid, from_epoch: int, to_epoch: int) -> int:
        curve_len = len(self.val_curves[uid])
        start     = min(from_epoch, curve_len)
        end       = min(to_epoch,   curve_len)
        return max(0, end - start)

    def _better(self, a: float, b: float) -> bool:
        return a < b if self.minimize else a > b

    # ── bracket ───────────────────────────────────────────────────────────────

    def _run_bracket(self, s, sampled_units, trained_up_to=None):
        # si no se pasa estado externo, comportamiento original (aislado)
        if trained_up_to is None:
            trained_up_to = defaultdict(int)

        r = self.R * (self.eta ** (-s))
        active = list(sampled_units)
        bracket_epochs = 0
        best_uid, best_loss = None, float("inf") if self.minimize else float("-inf")
        time_brancket = 0

        for i in range(s + 1):
            r_i = math.floor(r * (self.eta ** i))
            n_keep = max(1, math.floor(len(active) / self.eta))

            round_scores = []
            for uid in active:
                added = self._actual_epochs_added(uid, trained_up_to[uid], r_i)
                bracket_epochs += added
                trained_up_to[uid] += added
                time_brancket += added * self.unit_times[uid]

                score = self._score_at(uid, r_i)
                round_scores.append((uid, score))

                if self._better(score, best_loss):
                    best_loss, best_uid = score, uid

            round_scores.sort(key=lambda x: x[1], reverse=not self.minimize)
            active = [uid for uid, _ in round_scores[:n_keep]]

        return best_uid, best_loss, bracket_epochs, time_brancket

    # ── ejecución ─────────────────────────────────────────────────────────────

    def run_once(self, seed=None):
        rng = random.Random(seed)
        best_uid = None
        best_loss = float("inf") if self.minimize else float("-inf")
        epochs_used = 0
        global_trained_up_to = defaultdict(int)
        sampled_units_all = set()  # units vistos en esta run
        time_used = 0

        for s in range(self.s_max, -1, -1):
            n = math.ceil((self.s_max + 1) / (s + 1) * (self.eta ** s))
            sampled = (
                rng.sample(self.all_units, n)
                if n <= len(self.all_units)
                else rng.choices(self.all_units, k=n)
            )
            sampled_units_all.update(sampled)  # acumular units muestreados

            uid, score, ep, time_added = self._run_bracket(s, sampled, global_trained_up_to)
            epochs_used += ep
            time_used += time_added
            if self._better(score, best_loss):
                best_loss = score
                best_uid = uid

        # baseline: entrenar hasta el final todos los units muestreados
        total_epochs = sum(len(self.val_curves[u]) for u in sampled_units_all)
        real_best_loss = min([self.val_curves[u][-1] for u in sampled_units_all])
        total_time = sum(self.unit_times[u] for u in sampled_units_all)

        rank = sorted([self.val_curves[u][-1] for u in sampled_units_all])
        rank = [r >= best_loss for r in rank].index(True)

        time_score = (total_epochs - epochs_used) / total_epochs  # lower is better

        logging.info(f"Best loss: {best_loss}, Epochs saved: {time_score:0.2f}, rank: {rank}")

        performance_score = real_best_loss / best_loss  # lower is better

        score = np.sqrt(performance_score * time_score)

        saved_time = total_time - time_used

        return best_uid, best_loss, real_best_loss, epochs_used, total_epochs, score, rank, total_time, saved_time

    def run_montecarlo(self, n_runs=20, seed=42):
        return pd.DataFrame([
            {
                "run": i,
                "best_unit": uid,
                "filter_val_loss": val_loss,
                "best_val_loss": best_loss,
                "epochs_used": ep,
                "baseline_epochs": baseline,
                "val_score": val_score,
                "rank": rank,
                "total_time": total_time,
                "saved_time": saved_time
            }
            for i, (uid, val_loss, best_loss, ep, baseline, val_score, rank, total_time, saved_time) in enumerate(
                self.run_once(seed + i) for i in range(n_runs)
            )
        ])


# ══════════════════════════════════════════════════════════════════════════════
# Simulación
# ══════════════════════════════════════════════════════════════════════════════

def hyperband_simulation(config, ifold, queue, debug, directory, timeout):

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

        unit_times = {u: d['train__time'] for u, d in
                      hp_set.groupby('unit')[['train__time']].first().to_dict('index').items()}

        group_cols  = ["dataset", "task", "net"]
        eta_grid    = training_config.get('hyperband_eta_grid', [2, 3])
        n_runs      = training_config.get('hyperband_n_runs',   30)
        hb_seed     = training_config.get('hyperband_seed',     random_state)

        # ── paso 1: CV sobre train para seleccionar (R, eta) ─────────────────
        logging.info("Hyperband CV: optimizing R and eta")

        best = None
        R = None
        for eta in eta_grid:

            results = HyperbandSimulator.run_all_groups(
                fold_train, group_cols=group_cols, R=R, eta=eta,
                n_runs=n_runs, seed=hb_seed, unit_times_dict=unit_times,
            )
            mean_rp = results["val_score"].mean()
            if best is None or mean_rp < best["val_score"]:
                best = {"R": R, "eta": eta, "val_score": mean_rp}

                print(best)

        best_R   = best['R']
        best_eta = best['eta']

        # ── paso 2: Hyperband sobre train completo con los mejores params ─────
        logging.info("Hyperband: running on full train set")

        train_results = HyperbandSimulator.run_all_groups(
            train_df, group_cols=group_cols,
            R=best_R, eta=best_eta, n_runs=n_runs, seed=hb_seed,
            unit_times_dict=unit_times,
        )

        # ── paso 3: evaluación del ganador en test ────────────────────────────
        logging.info("Hyperband: running on test set with best params")

        test_results = HyperbandSimulator.run_all_groups(
            test_df, group_cols=group_cols,
            R=best_R, eta=best_eta, n_runs=n_runs, seed=hb_seed,
            unit_times_dict=unit_times,
        )

        # ── métricas globales y persistencia ─────────────────────────────────────

        mean_train_saved_pct = (train_results["epochs_saved"] / (train_results["epochs_saved"] +
                                                                 train_results["epochs_used"])).mean()
        mean_train_rank_pct = train_results["rank_pct"].mean()
        mean_train_val_score = train_results["val_score"].mean()
        train_mean_rank = train_results['rank'].mean()

        mean_test_saved_pct = (test_results["epochs_saved"] / (
                    test_results["epochs_saved"] + test_results["epochs_used"])).mean()
        mean_test_val_loss = test_results["filter_val_loss"].mean()
        mean_test_rank_pct = test_results["rank_pct"].mean()
        mean_test_val_score = test_results["val_score"].mean()
        filter_best_losses = list(test_results['filter_val_loss'].values)
        real_best_losses = list(test_results['best_val_loss'].values)
        train_total_time = test_results["total_time"].mean()
        test_saved_time = test_results["saved_time"].mean()

        rank_losses = list(test_results['rank'].values)
        test_results['experiment'] = test_results.dataset + "_" + test_results.task + "_" + test_results.net
        experiments = list(test_results.experiment.values)
        mean_epochs_saved = test_results["epochs_saved"].mean()
        mean_epochs_used = test_results["epochs_used"].mean()
        test_mean_rank = test_results['rank'].mean()
        total_time = test_results["total_time"].mean()
        saved_time = test_results["saved_time"].mean()
        saved_pct_per_exp = [s / t for s, t in zip(test_results["saved_time"], test_results["total_time"])]

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
            "epochs_used": mean_epochs_used,
            "test_epochs_saved_pct": mean_test_saved_pct,
            "train_epochs_saved_pct": mean_train_saved_pct,
            "mean_test_val_score": mean_test_val_score,
            "mean_train_val_score": mean_train_val_score,
            "test_mean_rank": test_mean_rank,
            "train_mean_rank": train_mean_rank,
            "random_state": random_state,
            "rank_losses": rank_losses,
            "experiments": experiments,
            "filter_best_losses": filter_best_losses,
            "real_best_losses": real_best_losses,
            "total_time": total_time,
            "saved_time": saved_time,
            "train_total_time": train_total_time,
            "test_saved_time": test_saved_time,
            "saved_pct_per_exp": saved_pct_per_exp,

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

