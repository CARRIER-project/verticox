from typing import Tuple, List, Dict

from sksurv.functions import StepFunction

from verticox.defaults import DEFAULT_KFOLD_SPLITS, DEFAULT_KFOLD_SEED
from verticox.node_manager import BaseNodeManager
from sklearn.model_selection import KFold
import numpy as np
import logging

logger = logging.getLogger(__name__)


def kfold_cross_validate(node_manager: BaseNodeManager,
                         n_splits=DEFAULT_KFOLD_SPLITS,
                         random_state=DEFAULT_KFOLD_SEED,
                         shuffle=True) -> \
        Tuple[List[float], List[Dict[str, float]], List[StepFunction]]:
    num_records = node_manager.num_total_records
    indices = np.arange(num_records)

    kf = KFold(n_splits, random_state=random_state, shuffle=shuffle)

    # We just want the selection of indices, not the full dataset.
    folds = kf.split(indices)

    c_indices = []
    coefs = []
    baseline_hazards = []
    for idx, (train_split, test_split) in enumerate(folds):
        logging.info(f'Training on fold {idx}')

        node_manager.reset(train_split)
        node_manager.fit()

        c_indices.append(node_manager.test())
        coefs.append(node_manager.coefs)

        baseline_hazards.append(node_manager.baseline_hazard)

    return c_indices, coefs, baseline_hazards
