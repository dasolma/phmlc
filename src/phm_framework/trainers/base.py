import logging
import sys
import time
import traceback
from collections import defaultdict

from phm_framework.logging import HASH_EXCLUDE, confighash, secure_decode, log_train, get_results
from phm_framework.trainers.utils import get_task
from phm_framework.utils import flat_dict

import pickle as pk
import numpy as np

logging.basicConfig(level=logging.INFO)

class BaseTrainer:

    def train(self, model_creator, config, ifold, queue, debug, directory, timeout):
        logging.info('Starting training (fold %d) %s' % (ifold, config))

        try:
            context = {}
            context['config'] = config
            context['config']['train']['fold'] = ifold

            training_config = config['train']
            net_config = config['model']
            data_config = config['data']

            net_name = net_config['net']
            data_name = data_config['dataset_name']
            data_target = data_config['dataset_target']

            task = get_task(data_name, data_target, model_creator)
            context['task_id'] = task

            csv_config = flat_dict(config.copy())

            csv_config = self.clean_config(csv_config)

            csv_config['train__fold'] = ifold
            nhash = confighash(csv_config, exclude=HASH_EXCLUDE)
            arch_hash = confighash(csv_config, exclude=HASH_EXCLUDE + ["train__fold"])
            csv_config['run_hash'] = nhash
            csv_config['arch_hash'] = arch_hash

            import os
            import tensorflow as tf
            from phm_framework import models
            from phm_framework.optimization import hyper_parameters as hp

            # prepare output directory
            if not os.path.exists(directory):
                os.makedirs(directory)

            net_history = f"{directory}/net_{net_name}_{data_name}_{task['target']}_{arch_hash}_{nhash}_history.pk"

            # if already train, return saved history
            previous_results = get_results(nhash, directory)
            if previous_results:
                queue.put((previous_results, arch_hash))
                return

            # data reading and prepare data generators
            logging.info("Reading data")
            ts_len = secure_decode(training_config, "ts_len", dtype=int, task=task)
            context['config']['train']['ts_len'] = ts_len

            preprocess = hp.PREPROCESS[secure_decode(data_config, "preprocess", str, default='norm', task=task)]()
            context['config']['data']['preprocess'] = preprocess

            normalize_output = (((task['type'] == "regression") and ('normalize_output' not in task)) or
                                (('normalize_output' in task) and (task['normalize_output'])))

            logging.info(f"Normalized output: {normalize_output}")

            context['config']['data']['normalize_output'] = normalize_output

            batch_size = secure_decode(training_config, "batch_size", dtype=int, task=task)
            context['config']['train']['batch_size'] = batch_size

            context['config']['data']['load_params'] = {
                "signal_length": context['config']['train']['ts_len'],
                "extract_features": context['config']['data']['extract_features'],
                "augmentation": context['config']['data']['augmentation'],
            }

            data = self.load_data(context)
            context['data'] = data

            logging.info("Finished Data reading")

            # create and compile model
            model_params = models.get_model_params(net_config, model_creator, task)

            if 'n_jobs' in model_params:
                model_params['n_jobs'] = training_config['n_jobs']
            if 'input_shape' in model_params:
                model_params['input_shape'] = self.get_input_shape(data['train'])
            if 'model_type' in model_params:
                model_params['model_type'] = f'classifier:{task["type"].split(":")[1]}' if 'classification' in task['type'] else 'regressor'
            if 'n_estimators' in model_params:
                model_params['n_estimators'] = training_config['iterations']


            metric_results = defaultdict(lambda : [])
            repetitions = secure_decode(training_config, "repetitions", dtype=int, task=task, default=1)

            random_states = np.random.randint(0, high=100000, size=(repetitions,))
            for rep in range(repetitions):
                if 'random_state' in model_params:
                    model_params['random_state'] = random_states[rep]

                csv_config.update(flat_dict({'model': model_params}))


                model = self.create_model(model_creator, model_params)

                logging.info("Model created")
                context['model'] = model

                metrics = hp.get_loss(task)
                val_gen = data['val']
                test_gen = data['test']
                if task['type'] == "regression":
                    val_gen10 = val_gen.loc[val_gen[task['target']] < 0.1]
                    val_gen0 = val_gen.loc[val_gen[task['target']] == 0]
                    test_gen10 = test_gen.loc[test_gen[task['target']] < 0.1]
                    test_gen0 = test_gen.loc[test_gen[task['target']] == 0]

                    if val_gen10.shape[0] > 0:
                        context['data']['val10'] = val_gen10
                    if val_gen0.shape[0] > 0:
                        context['data']['val_last'] = val_gen0
                    if test_gen10.shape[0] > 0:
                        context['data']['test10'] = test_gen10
                    if test_gen0.shape[0] > 0:
                        context['data']['test_last'] = test_gen0

                model.compile(context, metrics=metrics)

                # train
                start_time = time.time()
                results = model.fit(context)

                # save csv opt_history
                metric_results['train__time'].append((time.time() - start_time))
                for k in results.keys():
                    if k.startswith('val'):
                        metric_results[k].append(results[k])

                #csv_config.update({k: opt_history[k][-1] for k in opt_history.keys() if k.startswith('val')})

                test_metrics = self.evaluate_test(context)
                metric_results.update(test_metrics)


            for metric, values in metric_results.items():
                csv_config[metric] = np.mean(values)

            csv_config["train__status"] = "FINISHED"

            queue.put((results, arch_hash))

            log_train(csv_config, directory)

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

    def clean_config(self, config):
        del config['model__output']
        del config['model__output_dim']
        del config['data__extract_features']
        if 'train__stride' in config:
            del config['train__stride']
        del config['train__timeout']
        del config['train__epoch_timeout']

        return config

    def create_model(self, model_creator, model_params):
        model = model_creator(**model_params)
        return model

    def load_data(self, context):
        raise NotImplementedError()

    def get_input_shape(self, train_gen):
        sample = train_gen[0][0]
        if isinstance(sample, list):
            input_shape = sample[0].shape[1:]
        else:
            input_shape = sample.shape[1:]
        return input_shape

    def evaluate_test(self, context):
        results = {}
        for set_name in [k for k in context['data'].keys() if 'test' in k]:
            results.update(context['model'].evaluate(context, set=set_name, verbose=True))

        return results

