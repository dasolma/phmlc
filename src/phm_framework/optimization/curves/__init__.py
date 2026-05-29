from phmd import datasets
import numpy as np
from sklearn.model_selection import KFold
import random
import phm_framework as phmf

def load_curves(fold: int, num_folds: int = 5, normalize_output=False, filters=None, random_state=666,
                          test_dataset_names=[]):
    """
    Load and preprocess training, validation, and optionally test set generators for a given dataset and task.

    Parameters:
    -----------
    dataset_name : str
        The name of the dataset to load.
    sets : str
        Specifies which sets (e.g., 'train', 'val', 'test') to load from the dataset.
    task_name : str
        The specific task within the dataset, usually related to the target variable.
    fold : int
        The current fold index for cross-validation.
    num_folds : int, optional
        The total number of folds to use in cross-validation. Default is 5.
    batch_size : int, optional
        The number of samples per batch of computation. Default is 32.
    extra_channel : bool, optional
        If True, an extra channel will be added to the input data. Default is False.
    ts_len : int, optional
        The length of the time series to be used in the model. Default is 256.
    preprocess : callable, optional
        A preprocessing function or transformer (e.g., StandardScaler) that will be fitted and applied to the data.
    normalize_output : bool, optional
        If True, normalizes the output by unit, based on the task. Default is False.
    filters : list, optional
        Any additional filters to apply when loading the data. Default is None.
    random_state : int, optional
        Seed for random operations to ensure reproducibility. Default is 666.
    sequencer : class, optional
        The sequencer class to use for generating batches. Default is Sequence.
    test_pct : float, optional
        The percentage of data to be used for testing. Default is 0.3.

    Returns:
    --------
    sequencers : dict
        A dictionary containing the training, validation, and optionally the test set generators.
        Keys are 'train', 'val', and 'test' (if return_test is True).
    """

    dataset = "CURVES"
    task_name = "final_loss"

    # Read metadata for the specified dataset
    ds = datasets.Dataset(dataset)

    task = ds[task_name]  # Retrieve task-specific details
    task.folds = num_folds
    task.filters = filters
    task.normalize_output = normalize_output
    task.random_state = random_state

    # load
    X = task.load()[0]

    if 'train_loss' in X.columns:
        X.loc[X.train_loss > 8, 'train_loss'] = 8.0
        X['max_train_loss'] = X.groupby(['unit', 'dataset', 'task', 'net'])['val_loss'].transform('max')
        X['train_loss'] =  X['train_loss'] / X['max_train_loss']
        X.loc[X.train_loss > 1, 'train_loss'] = 1.0
        X['val_loss'] = X['val_loss'] / X['max_train_loss']
        del X['max_train_loss']
        X['num_epochs'] = X['num_epochs'] / 100
    else:
        activations = [a if isinstance(a, str) else a.__class__.__name__ for a in phmf.typing.ACTIVATIONS]
        rnn_cells = [a.__name__ for a in phmf.typing.RNN_CELLS]

        def get_code(x, elements):
            x = str(x)
            for i, a in enumerate(elements):
                if a in x:
                    return i

            return float(x)

        X['model__activation'] = X['model__activation'].map(
            lambda x: x if x is np.nan else round(get_code(x, activations)))
        X['model__dense_activation'] = X['model__dense_activation'].map(
            lambda x: x if x is np.nan else round(get_code(x, activations)))
        X['model__conv_activation'] = X['model__conv_activation'].map(
            lambda x: x if x is np.nan else round(get_code(x, activations)))
        X['model__cell_type'] = X['model__cell_type'].map(
            lambda x: x if x is np.nan else round(get_code(x, rnn_cells)))
        X['model__kernel_size'] = X['model__kernel_size'].map(
            lambda x: (x if x is np.nan else float(x)) if '(' not in str(x) else eval(x)[0] + eval(x)[1] / 100)
        X['model__batch_normalization'] = X['model__batch_normalization'].map(
            lambda x: x if x is np.nan else round(float(eval(str(x)))))
        X['model__bidirectional'] = X['model__bidirectional'].map(
            lambda x: x if x is np.nan else round(float(eval(str(x)))))

    if num_folds == 0:
        return X

    # split
    dataset_names = X.dataset.unique()

    if len(test_dataset_names) == 0:
        random.shuffle(dataset_names)
        test_end_index = int(len(dataset_names) * 0.2)
        test_dataset_names = dataset_names[:test_end_index]
        dataset_names = dataset_names[test_end_index:]
    else:
        dataset_names = np.array([d for d in dataset_names if d not in test_dataset_names])


    skf = KFold(n_splits=num_folds, random_state=random_state, shuffle=True)
    folds = list(skf.split(dataset_names))

    train_idx, val_idx = folds[fold]

    _sets = {}
    _sets['train'] = X[X.dataset.isin(dataset_names[train_idx])]
    _sets['val'] = X[X.dataset.isin(dataset_names[val_idx])]
    _sets['test'] = X[X.dataset.isin(test_dataset_names)]

    return _sets