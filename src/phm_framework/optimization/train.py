import argparse
import logging
import multiprocessing
import os, sys

from phm_framework.optimization.curves.fsldt import curves_fsldt

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


import itertools
from phmd import datasets
from phm_framework.optimization.curves.bohb import bohb_simulation
from phm_framework.optimization.curves.hyperband import hyperband_simulation
from phm_framework.logging import load_log, get_rows
import time

logging.basicConfig(level=logging.INFO)
logging.info("Working dir: " + os.getcwd())

from phm_framework.optimization.curves.train import (curves_train, arima_train, random_train, last_seen,
                                                     curves_fsl_train, curves_fsl_train_v2)

sem = multiprocessing.Semaphore(4)  # 👈 Máximo 4 en paralelo


def train_with_sem(config):
    with sem:
        print("Started training process")
        train(config)
        print("Finished training process")


def train_loop(lr):

    #learning_rate = [0.1, 0.01, 0.001, 0.0001]

    if args.model in ['rnn', 'rnn_cond']:
        nblocks = [1, 2, 3, 4]
        cells = [16, 32, 64, 128]
        bilstm = [True, False]

        for nblocks, cells, bilstm in itertools.product(nblocks, cells, bilstm):

            for ts_len in range(10, 2, -1):
                config = {
                    'model': {
                        'net': args.model,
                        'output_dim': hp.get_output_dim(task),
                        'output': "relu",
                        'nblocks': nblocks,
                        'bidirectional': bilstm,
                        'rnn_units': cells,
                    },

                    'data': {
                        'dataset_name': dataset,
                        'dataset_target': task_name,
                        'low_float_precision': True,
                        'preprocess': None,
                    },

                    'train': {
                        'epochs': 2 if args.debug else 100,
                        'batch_size': 32,
                        'timeout': 60 * 30,
                        'ts_len': 5 if args.debug else ts_len,
                        'lr': lr,
                        'verbose': True,
                        'num_folds': min(5, max_folds),
                        'random_state': random_state,
                        'stop_criteria': False,
                        'conditioning': args.features,
                        'debug': args.debug
                    },

                    'train_generator': {
                        'random_init': False,
                    },

                    'val_generator': {
                        'random_init': False,
                    },

                    'log': {
                        'directory': args.output,
                        'save_only_best': True
                    },

                    'train__stride': 0.,
                }

                train(config)

    elif args.model in ['protonet', 'protonetv2', 'fsldt']:
        nblocks = [1, 2, 3, 4]
        embedding_dims = [16, 32, 64, 128, 256]
        block_sizes = [1, 2, 3]

        processes = []
        for nblocks, embedding_dim, block_size in itertools.product(nblocks, embedding_dims, block_sizes):
                ts_len_range = [21] if args.model == 'fsldt' else range(5, 21, 5)
                for ts_len in ts_len_range:

                    config = {
                        'model': {
                            'net': args.model,
                            'output_dim': hp.get_output_dim(task),
                            'output': "relu",
                            'nblocks': nblocks,
                            'embedding_dim': embedding_dim,
                            'block_size': block_size,
                        },

                        'data': {
                            'dataset_name': dataset,
                            'dataset_target': task_name,
                            'low_float_precision': True,
                            'preprocess': None,
                        },

                        'train': {
                            'epochs': 1 if args.debug else 100,
                            'batch_size': 32,
                            'timeout': 60 * 30,
                            'ts_len': ts_len,
                            'lr': lr,
                            'verbose': True,
                            'num_folds': min(5, max_folds),
                            'random_state': random_state,
                            'stop_criteria': False,
                            'conditioning': args.features,
                            'debug': args.debug,
                            'use_current_train_curves': True,
                        },

                        'train_generator': {
                            'random_init': False,
                        },

                        'val_generator': {
                            'random_init': False,
                        },

                        'log': {
                            'directory': args.output,
                            'save_only_best': True
                        },

                        'train__stride': 0.,
                    }


                    p = multiprocessing.Process(target=train_with_sem, args=(config,))
                    p.start()
                    processes.append(p)
                    time.sleep(5)

        for p in processes:
            p.join()

    elif args.model in ['hb', 'bohb']:

        processes = []

        for ts_len in range(5, 21, 5):

            config = {
                'model': {
                    'net': args.model,
                    'output_dim': hp.get_output_dim(task),
                    'output': "relu",
                },

                'data': {
                    'dataset_name': dataset,
                    'dataset_target': task_name,
                    'low_float_precision': True,
                    'preprocess': None,
                },

                'train': {
                    'epochs': 1 if args.debug else 100,
                    'batch_size': 32,
                    'timeout': 60 * 30,
                    'ts_len': 9 if args.debug else ts_len,
                    'lr': lr,
                    'verbose': True,
                    'num_folds': min(5, max_folds),
                    'random_state': random_state,
                    'debug': args.debug,
                    'use_current_train_curves': True,
                },

                'train_generator': {
                    'random_init': False,
                },

                'val_generator': {
                    'random_init': False,
                },

                'log': {
                    'directory': args.output,
                    'save_only_best': True
                },

            }


            p = multiprocessing.Process(target=train, args=(config,))
            p.start()
            processes.append(p)
            time.sleep(5)

            #train(config)

        for p in processes:
            p.join()



    elif args.model == 'arima':

        for ts_len in range(7, 10):
            config = {
                'model': {
                    'net': args.model
                },

                'data': {
                    'dataset_name': dataset,
                    'dataset_target': task_name,
                    'low_float_precision': True,
                    'preprocess': None,
                },

                'train': {
                    'timeout': 60 * 30,
                    'ts_len': ts_len,
                    'verbose': True,
                    'num_folds': min(5, max_folds),
                    'random_state': 666,
                    'stop_criteria': False,
                    'monitor': 'test_mse'
                },

                'log': {
                    'directory': args.output,
                    'save_only_best': True
                },

            }

            train(config)

    elif args.model == 'last-seen':

        for ts_len in range(2, 10):
            config = {
                'model': {
                    'net': args.model
                },

                'data': {
                    'dataset_name': dataset,
                    'dataset_target': task_name,
                    'low_float_precision': True,
                    'preprocess': None,
                },

                'train': {
                    'timeout': 60 * 30,
                    'ts_len': ts_len,
                    'verbose': True,
                    'num_folds': min(5, max_folds),
                    'random_state': 666,
                    'stop_criteria': False,
                    'monitor': 'test_mse'
                },

                'log': {
                    'directory': args.output,
                    'save_only_best': True
                },

            }

            train(config)

    elif args.model == 'random':

        for random_pct in [0.8, 0.7, 0.6, 0.4, 0.3, 0.2]:
            config = {
                'model': {
                    'net': args.model
                },

                'data': {
                    'dataset_name': dataset,
                    'dataset_target': task_name,
                    'low_float_precision': True,
                    'preprocess': None,
                },

                'train': {
                    'timeout': 60 * 30,
                    'random_pct': random_pct,
                    'verbose': True,
                    'num_folds': min(5, max_folds),
                    'random_state': random_state,
                    'stop_criteria': False,
                    'monitor': 'test_mse'
                },

                'log': {
                    'directory': args.output,
                    'save_only_best': True
                },

            }

            train(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    model = [v for p, v in zip(sys.argv[:-1], sys.argv[1:]) if p == '-m' or p == '--model']
    if len(model) > 0:
        model = model[0]
    else:
        model = ""

    # Adding optional argument
    parser.add_argument("-m", "--model", help="Model params", type=str, required=True)
    parser.add_argument("-c", "--cuda", help="Cuda visible", choices=["0", "1"], default="", required=False)
    parser.add_argument("-nc", "--ncpus", help="CPUs to take", type=int, required=False, default=4)
    parser.add_argument("-b", "--debug", help="Debug mode", action='store_true')
    parser.add_argument("-o", "--output", help="Output dir", type=str, required=True)
    parser.add_argument("-r", "--random_state", help="Random seed", type=int, required=False)
    if model != "random" and model != "last-seen":
        parser.add_argument("-f", "--features", help="Features conditioning mode", type=str, required=False)
        parser.add_argument("-lr", "--lr", help="Learning rate", type=float, required=True)

    # Read arguments from command line
    args = parser.parse_args()

    logging.info("Params read")

    logging.info("GPU: " + str(args.cuda != ""))
    if args.cuda != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    ncpus = args.ncpus
    logging.info(f"Limiting to tensorflow to use only {ncpus} threads")

    os.environ["OMP_NUM_THREADS"] = str(ncpus)
    os.environ["NUMEXPR_MAX_THREADS"] = str(ncpus)
    os.environ["NUMEXPR_NUM_THREADS"] = str(ncpus)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(ncpus)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(ncpus)

    import tensorflow as tf

    tf.config.threading.set_inter_op_parallelism_threads(
        ncpus
    )
    tf.config.threading.set_intra_op_parallelism_threads(
        ncpus
    )
    tf.config.set_soft_device_placement(True)

    os.environ['RAY_memory_monitor_refresh_ms'] = "0"

    import phm_framework
    from phm_framework.optimization import hyper_parameters as hp

    def train(config):


        for key in config.keys():
            if '__' in key:
                sect, param = key.split('__')
                config[sect][param] = config[key]

        config = {k: v for k, v in config.items() if '__' not in k}
        print(config)

        if args.model == 'random':
            return phm_framework.optimization.utils.parameter_opt_cv(
                None,
                config,
                trainer=random_train,
                debug=args.debug,
            )
        elif args.model == 'arima':
            return phm_framework.optimization.utils.parameter_opt_cv(
                None,
                config,
                trainer=arima_train,
                debug=args.debug,
            )
        elif args.model == 'last-seen':
            return phm_framework.optimization.utils.parameter_opt_cv(
                None,
                config,
                trainer=last_seen,
                debug=args.debug,
            )

        elif args.model == 'protonet':
                creator = get_model_creator('protonet')

                return phm_framework.optimization.utils.parameter_opt_cv(
                    creator,
                    config,
                    trainer=curves_fsl_train,
                    debug=args.debug
                )

        elif args.model == 'protonetv2':
            config['model']['net'] = 'protonet'
            creator = get_model_creator('protonet')

            return phm_framework.optimization.utils.parameter_opt_cv_v2(
                creator,
                config,
                trainer=curves_fsl_train_v2,
                debug=args.debug
            )

        elif args.model == 'fsldt':
            config['model']['net'] = 'fsldt'
            creator = get_model_creator('protonet')

            return phm_framework.optimization.utils.parameter_opt_cv_fsldt(
                creator,
                config,
                trainer=curves_fsldt,
                debug=args.debug
            )

        elif args.model == 'hb':
            config['model']['net'] = 'hb'

            return phm_framework.optimization.utils.parameter_opt_cv_hb(
                None,
                config,
                trainer=hyperband_simulation,
                debug=args.debug
            )

        elif args.model == 'bohb':
            config['model']['net'] = 'bohb'

            return phm_framework.optimization.utils.parameter_opt_cv_hb(
                None,
                config,
                trainer=bohb_simulation,
                debug=args.debug
            )


    def get_model_creator(net=None):
        net_creator_func = f"create_model"
        net_module = getattr(getattr(phm_framework, 'models'), net or args.model)
        creator = getattr(net_module, net_creator_func)

        return creator

    dataset = "CURVES"
    task_name = "final_loss"
    ds = datasets.Dataset(dataset)
    task = ds[task_name].meta

    max_folds = 1 if args.model == 'arima' else 3

    if args.random_state:
        random_state = args.random_state
        train_loop(args.lr)

    else:

        isnet = lambda x: x in ['rnn', 'rnn_cond', 'protonet', 'protonetv2', 'fsldt', 'hb', 'bohb']
        if isnet(args.model):
            random_states = [29, 8162, 1391, 2821, 3709, 106, 4665, 7204, 6321, 8444]

            for random_state in random_states:
                train_loop(args.lr)

        elif args.model == 'arima' or args.model == 'last-seen':
            train_loop(None)

        elif args.model == 'random':
            random_states = [29, 8162, 1391, 2821, 3709, 106, 4665, 7204, 6321, 8444]

            for random_state in random_states:
                train_loop(None)

