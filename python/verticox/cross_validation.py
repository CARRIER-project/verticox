from verticox.node_manager import BaseNodeManager
from sklearn.model_selection import KFold
import numpy as np
import logging

logger = logging.getLogger(__name__)


_DEFAULT_NUM_SPLIT = 5
_DEFAULT_SEED = 0


def kfold_cross_validate(node_manager: BaseNodeManager,
                         n_splits=_DEFAULT_NUM_SPLIT,
                         random_state=_DEFAULT_SEED,
                         shuffle=True):
    num_records = node_manager.num_total_records
    indices = np.arange(num_records)

    kf = KFold(n_splits, random_state=random_state, shuffle=shuffle)

    # We just want the selection of indices, not the full dataset.
    folds = kf.split(indices)

    c_indices = []
    coefs = []

    for idx, (train_split, test_split) in enumerate(folds):
        logging.info(f'Training on fold {idx}')

        node_manager.reset(train_split)
        node_manager.fit()

        c_indices.append(node_manager.test())
        coefs.append(node_manager.coefs)

    return c_indices, coefs
