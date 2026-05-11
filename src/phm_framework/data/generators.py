from collections import defaultdict
from collections.abc import Iterable
import tensorflow as tf
import random
import logging
import numpy as np
import phmd
from phmd import datasets
from sklearn.model_selection import KFold


def _tuple(x):

    if isinstance(x, Iterable):
        return tuple(x)
    else:
        return x


class Sequence(tf.keras.utils.Sequence):

    def __init__(self, data, unit_cols, features_cols, target_col,  batches_per_epoch=1000, batch_size=32,
                 extra_channel=False, ts_len=256, ts_consider=1., stride=1, random_init=True):

        if data is None:
            return

        self.data = data
        self.stride = stride
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.target_col = target_col
        self.units = [list(l) for l in self.data[unit_cols].drop_duplicates().values]
        self.random_init = random_init

        self.ts_consider = ts_consider

        units = str(random.sample(self.units, min(len(self.units), 20))).replace('\n', ',')
        logging.info(f"Units: {units}. Total units: {len(self.units)}")

        self.data = {}
        self.extra_channel = extra_channel
        self.ts_len = ts_len

        logging.info("Indexing features by unit")
        units = data[unit_cols].values
        self.data = data.groupby(unit_cols)
        self.units = [x for x in self.data.groups]
        if len(unit_cols) == 1:
            self.target = {_tuple(x): self.data.get_group(x)[target_col].values for x in self.units}
            self.data = {_tuple(x): self.data.get_group(x)[features_cols].values for x in self.units}
        else:
            self.target = {_tuple(x): self.data.get_group(x)[target_col].values for x in self.units}
            self.data = {_tuple(x): self.data.get_group(x)[features_cols].values for x in self.units}


        self.feature_cols = features_cols
        self.nfeatures = len(features_cols)

    def clone(self):
        seq = Sequence(None, None, None, None)
        seq.data = self.data
        seq.batch_size = self.batch_size
        seq.batches_per_epoch = self.batches_per_epoch
        seq.units = self.units
        seq.ts_consider = self.ts_consider
        seq.extra_channel = self.extra_channel
        seq.ts_len = self.ts_len
        seq.target = self.target
        seq.target_col = self.target_col
        seq.stride = self.stride
        seq.nfeatures = self.nfeatures
        seq.random_init = self.random_init

        return seq

    def __len__(self):
        return self.batches_per_epoch


    def clone(self):
        seq = Sequence(None, None, None, None)

        seq.data = self.data
        seq.batch_size = self.batch_size
        seq.batches_per_epoch = self.batches_per_epoch
        seq.units = self.units
        seq.ts_consider = self.ts_consider
        seq.extra_channel = self.extra_channel
        seq.ts_len = self.ts_len
        seq.target = self.target
        seq.target_col = self.target_col
        seq.feature_cols = self.feature_cols
        seq.stride = self.stride
        seq.nfeatures = self.nfeatures
        seq.random_init = self.random_init

        return seq

    def __getitem__(self, idx):
        D = self.data
        ts_len = self.ts_len

        # initialize feature matrix
        X, Y = self.init_batch_matrices(ts_len)

        # create batch
        for i in range(self.batch_size):
            # select a random unit
            unit = self.units[random.randint(0, len(self.units) - 1)]

            self.generate_sample(X, Y, i, ts_len, unit)

        if isinstance(X, list):
            X = [x.astype('float32') for x in X]
        else:
            X = X.astype('float32')

        return X, Y.astype('float32')

    def init_batch_matrices(self, ts_len):
        X = np.zeros(shape=(self.batch_size, ts_len, self.nfeatures))
        Y = np.zeros(shape=(self.batch_size,))

        return X, Y

    def generate_sample(self, X, Y, i, ts_len, unit):

        # get unit data
        Db = self.data[_tuple(unit)]
        T = self.target[_tuple(unit)]
        # select random point
        assert Db.shape[0] >= ts_len
        L = Db.shape[0]
        if self.ts_consider == 0:
            k = max(0, L - (ts_len * self.stride) - 1)
        else:
            Lini = min(int(L * (1 - self.ts_consider)), L - (ts_len * self.stride) - 1)
            k = max(0, random.randint(Lini, L - (ts_len * self.stride) - 1))
        indexes = np.arange(k, k + (ts_len * self.stride), self.stride)
        # to avoid out of index
        indexes = np.clip(indexes, 0, L - 1)

        self.update_batch_matrices(Db, T, X, Y, i, indexes, k, ts_len, unit)

    def update_batch_matrices(self, Db, T, X, Y, i, indexes, k, ts_len, unit):
        if self.extra_channel:
            X[i, :, :, 0] = Db[indexes, :]
        else:
            X[i, :, :] = Db[indexes, :]
        # select last point of sequence target
        Y[i] = T[k + ts_len]


class SequenceV2(tf.keras.utils.Sequence):

    def __init__(self, data, unit_cols, features_cols, target_col,  batches_per_epoch=1000, batch_size=32,
                 extra_channel=False, ts_len=256, ts_consider=1., stride=1, random_init=True, shots=100):

        if data is None:
            return

        self.data = data
        self.stride = stride
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.target_col = target_col
        self.units = [list(l) for l in self.data[unit_cols].drop_duplicates().values]
        self.random_init = random_init

        self.ts_consider = ts_consider

        units = str(random.sample(self.units, min(len(self.units), 20))).replace('\n', ',')
        logging.info(f"Units: {units}. Total units: {len(self.units)}")

        self.data = {}
        self.extra_channel = extra_channel
        self.ts_len = ts_len
        self._lens = defaultdict(lambda: [])

        logging.info("Indexing features by unit")

        self.data = data.groupby(unit_cols)
        self.units = [x for x in self.data.groups]
        if len(unit_cols) == 1:
            self.target = {_tuple(x): self.data.get_group(x)[target_col].values for x in self.units}
            self.data = {_tuple(x): self.data.get_group(x)[features_cols].values for x in self.units}
        else:
            self.target = {_tuple(x): self.data.get_group(x)[target_col].values for x in self.units}
            self.data = {_tuple(x): self.data.get_group(x)[features_cols].values for x in self.units}

        for unit, _size in data.groupby(unit_cols).size().to_dict().items():
            for s in range(1, _size):
                self._lens[s].append(_tuple(unit))

        self._valid_lens = [k for k, v in self._lens.items() if 2 * shots < len(v) and k <= ts_len]

        self.feature_cols = features_cols
        self.nfeatures = len(features_cols)
        self.shots = shots

        self.ts_len = max(ts_len, 20)


    def __len__(self):
        return self.batches_per_epoch


    def clone(self):
        seq = SequenceV2(None, None, None, None)

        seq.data = self.data
        seq.batch_size = self.batch_size
        seq.batches_per_epoch = self.batches_per_epoch
        seq.units = self.units
        seq.ts_consider = self.ts_consider
        seq.extra_channel = self.extra_channel
        seq.ts_len = self.ts_len
        seq.target = self.target
        seq.target_col = self.target_col
        seq.feature_cols = self.feature_cols
        seq.stride = self.stride
        seq.nfeatures = self.nfeatures
        seq.random_init = self.random_init

        return seq

    def __getitem__(self, idx):

        ts_len = random.choice(self._valid_lens)

        # initialize feature matrix
        X, S, Y, SY = self.init_batch_matrices(ts_len)

        if self.ts_len > ts_len:
            mask = np.vstack((np.ones((ts_len, 1)), np.vstack(np.zeros((self.ts_len - ts_len, 1)))))
        else:
            mask = np.ones((ts_len, 1))

        # create batch
        for i in range(self.batch_size):
            # select a random unit
            unit = random.choice(self._lens[ts_len])

            # query sample
            X[i, :, :], Y[i] = self.get_curve_and_target(unit, ts_len, mask)

        # support samples
        idx = np.arange(len(self._lens[ts_len]))
        np.random.shuffle(idx)
        sunits = [self._lens[ts_len][i] for i in idx[:self.shots]]

        for j, unit in enumerate(sunits):
            S[j, :, :], SY[j] = self.get_curve_and_target(unit, ts_len, mask)

        return X, Y.astype('float32')

    def get_curve_and_target(self, unit, ts_len, mask):

        return (np.hstack((np.vstack((self.data[_tuple(unit)][:ts_len], np.zeros((self.ts_len - ts_len, 2)))), mask)),
                self.target[_tuple(unit)][-1])


    def init_batch_matrices(self, ts_len):
        X = np.zeros(shape=(self.batch_size, self.ts_len, self.nfeatures + 1))
        S = np.zeros(shape=(self.shots, self.ts_len, self.nfeatures + 1))
        Y = np.zeros(shape=(self.batch_size,))
        SY = np.zeros(shape=(self.shots,))
        return X, S, Y, SY

    def generate_sample(self, X, Y, i, ts_len, unit):

        # get unit data
        Db = self.data[_tuple(unit)]
        T = self.target[_tuple(unit)]
        # select random point
        assert Db.shape[0] >= ts_len
        L = Db.shape[0]
        if self.ts_consider == 0:
            k = max(0, L - (ts_len * self.stride) - 1)
        else:
            Lini = min(int(L * (1 - self.ts_consider)), L - (ts_len * self.stride) - 1)
            k = max(0, random.randint(Lini, L - (ts_len * self.stride) - 1))
        indexes = np.arange(k, k + (ts_len * self.stride), self.stride)
        # to avoid out of index
        indexes = np.clip(indexes, 0, L - 1)

        self.update_batch_matrices(Db, T, X, Y, i, indexes, k, ts_len, unit)

    def update_batch_matrices(self, Db, T, X, Y, i, indexes, k, ts_len, unit):
        X[i, :, :] = Db[indexes, :]
        # select last point of sequence target
        Y[i] = T[k + ts_len]


class FSLSequence(tf.keras.utils.Sequence):

    def __init__(self, data, unit_cols, features_cols, target_col,  batches_per_epoch=1000, batch_size=32,
                 extra_channel=False, ts_len=256, ts_consider=1., stride=1, random_init=True, shots=100):

        if data is None:
            return

        self.data = data
        self.stride = stride
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.target_col = target_col
        self.units = [list(l) for l in self.data[unit_cols].drop_duplicates().values]
        self.random_init = random_init

        self.ts_consider = ts_consider

        units = str(random.sample(self.units, min(len(self.units), 20))).replace('\n', ',')
        logging.info(f"Units: {units}. Total units: {len(self.units)}")

        self.data = {}
        self.extra_channel = extra_channel
        self.ts_len = ts_len
        self._lens = defaultdict(lambda: [])

        logging.info("Indexing features by unit")

        self.data = data.groupby(unit_cols)
        self.units = [x for x in self.data.groups]
        if len(unit_cols) == 1:
            self.target = {_tuple(x): self.data.get_group(x)[target_col].values for x in self.units}
            self.data = {_tuple(x): self.data.get_group(x)[features_cols].values for x in self.units}
        else:
            self.target = {_tuple(x): self.data.get_group(x)[target_col].values for x in self.units}
            self.data = {_tuple(x): self.data.get_group(x)[features_cols].values for x in self.units}

        for unit, _size in data.groupby(unit_cols).size().to_dict().items():
            for s in range(1, _size):
                self._lens[s].append(_tuple(unit))

        self._valid_lens = [k for k, v in self._lens.items() if 2 * shots < len(v) and k <= ts_len]

        self.feature_cols = features_cols
        self.nfeatures = len(features_cols)
        self.shots = shots

        self.ts_len = max(ts_len, 20)


    def __len__(self):
        return self.batches_per_epoch


    def clone(self):
        seq = FSLSequence(None, None, None, None)

        seq.data = self.data
        seq.batch_size = self.batch_size
        seq.batches_per_epoch = self.batches_per_epoch
        seq.units = self.units
        seq.ts_consider = self.ts_consider
        seq.extra_channel = self.extra_channel
        seq.ts_len = self.ts_len
        seq.target = self.target
        seq.target_col = self.target_col
        seq.feature_cols = self.feature_cols
        seq.stride = self.stride
        seq.nfeatures = self.nfeatures
        seq.random_init = self.random_init

        return seq

    def __getitem__(self, idx):

        ts_len = random.choice(self._valid_lens)

        # initialize feature matrix
        X, S, Y, SY = self.init_batch_matrices(ts_len)

        if self.ts_len > ts_len:
            mask = np.vstack((np.ones((ts_len, 1)), np.vstack(np.zeros((self.ts_len - ts_len, 1)))))
        else:
            mask = np.ones((ts_len, 1))

        # create batch
        for i in range(self.batch_size):
            # select a random unit
            unit = random.choice(self._lens[ts_len])

            # query sample
            X[i, :, :], Y[i] = self.get_curve_and_target(unit, ts_len, mask)

        # support samples
        idx = np.arange(len(self._lens[ts_len]))
        np.random.shuffle(idx)
        sunits = [self._lens[ts_len][i] for i in idx[:self.shots]]

        for j, unit in enumerate(sunits):
            S[j, :, :], SY[j] = self.get_curve_and_target(unit, ts_len, mask)

        return (X, S, SY.astype('float32')), (Y.astype('float32'))

    def get_curve_and_target(self, unit, ts_len, mask):

        return (np.hstack((np.vstack((self.data[_tuple(unit)][:ts_len], np.zeros((self.ts_len - ts_len, 2)))), mask)),
                self.target[_tuple(unit)][-1])


    def init_batch_matrices(self, ts_len):
        X = np.zeros(shape=(self.batch_size, self.ts_len, self.nfeatures + 1))
        S = np.zeros(shape=(self.shots, self.ts_len, self.nfeatures + 1))
        Y = np.zeros(shape=(self.batch_size,))
        SY = np.zeros(shape=(self.shots,))
        return X, S, Y, SY

    def generate_sample(self, X, Y, i, ts_len, unit):

        # get unit data
        Db = self.data[_tuple(unit)]
        T = self.target[_tuple(unit)]
        # select random point
        assert Db.shape[0] >= ts_len
        L = Db.shape[0]
        if self.ts_consider == 0:
            k = max(0, L - (ts_len * self.stride) - 1)
        else:
            Lini = min(int(L * (1 - self.ts_consider)), L - (ts_len * self.stride) - 1)
            k = max(0, random.randint(Lini, L - (ts_len * self.stride) - 1))
        indexes = np.arange(k, k + (ts_len * self.stride), self.stride)
        # to avoid out of index
        indexes = np.clip(indexes, 0, L - 1)

        self.update_batch_matrices(Db, T, X, Y, i, indexes, k, ts_len, unit)

    def update_batch_matrices(self, Db, T, X, Y, i, indexes, k, ts_len, unit):
        X[i, :, :] = Db[indexes, :]
        # select last point of sequence target
        Y[i] = T[k + ts_len]

def load_train_generators(dataset_name: str, task_name, fold: int, num_folds: int = 5,
                          batch_size: int = 32, extra_channel: bool = False, ts_len: int = 256, preprocess=None,
                          return_train_val=True, return_test=False, normalize_output=False, filters=None,
                          random_state=666, sequencer=Sequence, test_pct=0.3):
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
    return_train_val : bool, optional
        If True, the function will return the training and validation set generators. Default is True.
    return_test : bool, optional
        If True, the function will return the test set generator as well. Default is False.
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

    # Read metadata for the specified dataset
    ds = datasets.Dataset(dataset_name)

    task = ds[task_name]  # Retrieve task-specific details
    task.folds = num_folds
    task.filters = filters
    task.normalize_output = normalize_output
    task.random_state = random_state

    # Load the training, validation, and test sets
    _sets = task[fold]

    sequencers = {}  # Initialize the dictionary to hold the data generators

    # Create data generators for training and validation sets
    if return_train_val:
        sequencers['train'] = sequencer(
            _sets['train'], task.meta['identifier'], task.meta['features'], task.meta['target'],
            batches_per_epoch=1000, batch_size=batch_size, extra_channel=extra_channel, ts_len=ts_len
        )
        sequencers['val'] = sequencer(
            _sets['val'], task.meta['identifier'], task.meta['features'], task.meta['target'], batches_per_epoch=100,
            batch_size=batch_size, extra_channel=extra_channel, ts_len=ts_len, ts_consider=1
        )

    # Create a data generator for the test set, if requested
    if return_test:
        if len(_sets) > 2:  # Ensure the test set is available
            sequencers['test'] = sequencer(
                _sets['test'], task.meta['identifier'], task.meta['features'], task.meta['target'], batches_per_epoch=1000,
                batch_size=batch_size, extra_channel=extra_channel, ts_len=ts_len, ts_consider=1
            )

    return sequencers  # Return the dictionary of data generators


def load_train_net_generators_v2(dataset_name: str, task_name, fold: int, num_folds: int = 5,
                          batch_size: int = 32, extra_channel: bool = False, ts_len: int = 256,
                          normalize_output=False, filters=None, random_state=666, sequencer=Sequence,
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

    # Read metadata for the specified dataset
    ds = datasets.Dataset(dataset_name)

    task = ds[task_name]  # Retrieve task-specific details
    task.folds = num_folds
    task.filters = filters
    task.normalize_output = normalize_output
    task.random_state = random_state

    # load
    X = task.load()[0]
    X.loc[X.train_loss > 8, 'train_loss'] = 8.0
    X['max_train_loss'] = X.groupby(['unit', 'dataset', 'task', 'net'])['val_loss'].transform('max')
    X['train_loss'] =  X['train_loss'] / X['max_train_loss']
    X.loc[X.train_loss > 1, 'train_loss'] = 1.0
    X['val_loss'] = X['val_loss'] / X['max_train_loss']
    del X['max_train_loss']
    X['num_epochs'] = X['num_epochs'] / 100

    # split
    dataset_names = X.dataset.unique()

    if len(dataset_names) == 0:
        random.shuffle(dataset_names)
        train_end_index = int(len(dataset_names) * 0.4)
        dataset_names = dataset_names[:train_end_index]
    else:
        dataset_names = np.array([dn for dn in dataset_names if dn not in test_dataset_names])

    skf = KFold(n_splits=num_folds, random_state=random_state, shuffle=True)
    folds = list(skf.split(dataset_names))

    train_idx, val_idx = folds[fold]

    _sets = {}
    _sets['train'] = X[X.dataset.isin(dataset_names[train_idx])]
    _sets['val'] = X[X.dataset.isin(dataset_names[val_idx])]
    # todo: eliminar [:x]
    test_datasetnames = list(filter(lambda n:  n not in dataset_names, X.dataset.unique()))
    _sets['test'] = X[X.dataset.isin(test_datasetnames)]

    logging.info(f"Test shape: {_sets['test'].shape}")

    sequencers = {}  # Initialize the dictionary to hold the data generators

    sequencers['train'] = sequencer(
        _sets['train'], task.meta['identifier'], task.meta['features'], task.meta['target'],
        batches_per_epoch=1000, batch_size=batch_size, extra_channel=extra_channel, ts_len=ts_len
    )
    sequencers['val'] = sequencer(
        _sets['val'], task.meta['identifier'], task.meta['features'], task.meta['target'], batches_per_epoch=100,
        batch_size=batch_size, extra_channel=extra_channel, ts_len=ts_len, ts_consider=1
    )

    sequencers['test'] = sequencer(
        _sets['test'], task.meta['identifier'], task.meta['features'], task.meta['target'], batches_per_epoch=1000,
        batch_size=batch_size, extra_channel=extra_channel, ts_len=ts_len, ts_consider=1
    )

    return sequencers  # Return the dictionary of data generators
