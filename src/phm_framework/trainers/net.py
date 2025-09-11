import logging
import time

from phm_framework.data.generators import load_train_generators
from phm_framework.logging import secure_decode
from phm_framework.models.utils import TimeStopping, AdditionalRULValidationSets
from phm_framework.trainers.base import BaseTrainer
import tensorflow as tf

logging.basicConfig(level=logging.INFO)


class NetWrapper:

    def __init__(self, model):
        self.model = model

    def compile(self, context, metrics):
        lr = context['config']['train']['lr']
        loss = metrics[0]
        metrics = metrics[1:]
        return self.model.compile(loss=loss, metrics=metrics,
                                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                                  run_eagerly=False
                                  )

    def fit(self, context, *args, **kwargs):
        train_gen = context['data']['train']
        val_gen = context['data']['val']
        epochs = context['config']['train']['iterations']
        monitor = context['config']['train']['monitor']
        task = context['task_id']

        epoch_timeout = secure_decode(context['config']['train'], "epoch_timeout", int, default=None, task=task)
        timeout = secure_decode(context['config']['train'], "timeout", int, default=None, task=task)
        verbose = secure_decode(context['config']['train'], "verbose", bool, default=False, task=task)

        es = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=8)
        rlr = tf.keras.callbacks.ReduceLROnPlateau(patience=3)
        ts = TimeStopping(seconds=timeout, epoch_seconds=epoch_timeout, verbose=1)

        extra_callbacks = []
        if task['type'] == "regression":
            val_gen10 = val_gen.clone()
            val_gen10.ts_consider = 0.1

            val_gen0 = val_gen.clone()
            val_gen0.ts_consider = 0

            extra_callbacks.append(AdditionalRULValidationSets([(val_gen0, 'val_last'), (val_gen10, 'val10')]))

        class_weights = None
        if "classification" in task['type']:
            class_weights = any([td < (1 / len(task['target_distribution'])) * 0.7 for td in task['target_distribution']])

            if class_weights:
                class_weights = {c: (task['num_units'] / len(task['target_distribution'])) * (1 / (td * task['num_units'])) for c, td in enumerate(task['target_distribution'])}
            else:
                class_weights = None

        self.model.summary(print_fn=lambda x: logging.info(x))

        logging.info("Started training")
        start_time = time.time()
        history = self.model.fit(train_gen, validation_data=val_gen,
                            epochs=epochs, verbose=(2 if verbose else 0),
                            callbacks=[es, rlr, ts] + extra_callbacks,
                            class_weight=class_weights)
        history = history.history

        results = {}
        results['train__time'] = (time.time() - start_time)
        results.update({k: history[k][-1] for k in history.keys() if k.startswith('val')})

        print(', '.join(f"{k}: {v:0.4f}" for k, v in results.items()))

        return results

    def evaluate(self, context, set='test', verbose=True):
        gen = context['data'][set]
        task = context['task_id']

        logging.info(f"Evaluating on {set} set")

        results = {}
        test_metrics = self.model.evaluate(gen, verbose=(2 if verbose else 0))
        for i, metric_name in enumerate(self.model.metrics_names):
            results[f"test_{metric_name}"] = test_metrics[i]

        if task['type'] == "regression":
            test_gen10 = gen.clone()
            test_gen10.ts_consider = 0.1

            test_metrics = self.model.evaluate(test_gen10, verbose=(2 if verbose else 0))
            for i, metric_name in enumerate(self.model.metrics_names):
                results[f"test10_{metric_name}"] = test_metrics[i]

            test_gen0 = gen.clone()
            test_gen0.ts_consider = 0

            test_metrics = self.model.evaluate(test_gen0, verbose=(2 if verbose else 0))
            for i, metric_name in enumerate(self.model.metrics_names):
                results[f"test_last_{metric_name}"] = test_metrics[i]


        if verbose:
            print(', '.join(f"{k}: {v:0.4f}" for k, v in results.items()))

        return results


class NetTrainer(BaseTrainer):

    def create_model(self, *args, **kwargs):
        model = super().create_model(*args, **kwargs)

        return NetWrapper(model)

    def load_data(self, context):
        data_name = context['config']['data']['dataset_name']
        task_name = context['config']['data']['dataset_target']
        ifold = context['config']['train']['fold']
        num_folds = context['config']['train']['num_folds']
        preprocess = context['config']['data']['preprocess']
        augmentation = context['config']['data']['augmentation']
        task = context['task_id']
        ts_len = context['config']['train']['ts_len']
        normalize_output = context['config']['data']['normalize_output']


        data = load_train_generators(data_name, task_name=task_name,
            ts_len=ts_len, fold=ifold, num_folds=num_folds, preprocess=preprocess, return_test=True,
            normalize_output=normalize_output)

        return data