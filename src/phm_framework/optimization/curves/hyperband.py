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
        self.all_units  = list(val_curves.keys())
        self.R          = R or int(max(len(c) for c in val_curves.values()))
        self.eta        = eta
        self.minimize   = minimize
        self.s_max      = math.floor(math.log(self.R, self.eta))

    @classmethod
    def from_group(
        cls,
        group_df: pd.DataFrame,
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
        return cls(val_curves, R=R, eta=eta, minimize=minimize)

    @classmethod
    def run_all_groups(
        cls,
        df: pd.DataFrame,
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
                sim        = cls.from_group(group_df, R=R, eta=eta, minimize=minimize)
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
            "rank", "rank_pct",
            "epochs_used", "epochs_saved",
        ]
        result = pd.concat(records, ignore_index=True)
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
        n_units = len(self.all_units)
        rankings = self.unit_rankings()

        mc_df = mc_df.copy()
        mc_df["rank"] = mc_df["best_unit"].map(rankings)
        mc_df["rank_pct"] = (mc_df["rank"] - 1) / max(n_units - 1, 1)
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
        best_uid, best_score = None, float("inf") if self.minimize else float("-inf")

        for i in range(s + 1):
            r_i = math.floor(r * (self.eta ** i))
            n_keep = max(1, math.floor(len(active) / self.eta))

            round_scores = []
            for uid in active:
                added = self._actual_epochs_added(uid, trained_up_to[uid], r_i)
                bracket_epochs += added
                trained_up_to[uid] += added

                score = self._score_at(uid, r_i)
                round_scores.append((uid, score))

                if self._better(score, best_score):
                    best_score, best_uid = score, uid

            round_scores.sort(key=lambda x: x[1], reverse=not self.minimize)
            active = [uid for uid, _ in round_scores[:n_keep]]

        return best_uid, best_score, bracket_epochs

    # ── ejecución ─────────────────────────────────────────────────────────────

    def run_once(self, seed=None):
        rng = random.Random(seed)
        global_best_uid = None
        global_best_score = float("inf") if self.minimize else float("-inf")
        total_epochs = 0
        global_trained_up_to = defaultdict(int)
        sampled_units_all = set()  # units vistos en esta run

        for s in range(self.s_max, -1, -1):
            n = math.ceil((self.s_max + 1) / (s + 1) * (self.eta ** s))
            sampled = (
                rng.sample(self.all_units, n)
                if n <= len(self.all_units)
                else rng.choices(self.all_units, k=n)
            )
            sampled_units_all.update(sampled)  # acumular units muestreados

            uid, score, ep = self._run_bracket(s, sampled, global_trained_up_to)
            total_epochs += ep
            if self._better(score, global_best_score):
                global_best_score = score
                global_best_uid = uid

        # baseline: entrenar hasta el final todos los units muestreados
        baseline_epochs = sum(len(self.val_curves[u]) for u in sampled_units_all)

        return global_best_uid, global_best_score, total_epochs, baseline_epochs

    def run_montecarlo(self, n_runs=20, seed=42):
        return pd.DataFrame([
            {
                "run": i,
                "best_unit": uid,
                "best_val_loss": score,
                "epochs_used": ep,
                "baseline_epochs": baseline,  # ← nuevo
            }
            for i, (uid, score, ep, baseline) in enumerate(
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

            results = HyperbandSimulator.run_all_groups(
                fold_train, group_cols=group_cols, R=R, eta=eta,
                n_runs=n_runs, seed=hb_seed,
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

        train_results = HyperbandSimulator.run_all_groups(
            train_df, group_cols=group_cols,
            R=best_R, eta=best_eta, n_runs=n_runs, seed=hb_seed,
        )

        # ── paso 3: evaluación del ganador en test ────────────────────────────
        logging.info("Hyperband: running on test set with best params")

        test_results = HyperbandSimulator.run_all_groups(
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

