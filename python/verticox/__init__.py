# Suppress numba debug logging
import logging
import sys

logging.basicConfig(stream=sys.stdout)
logging.getLogger("numba").setLevel(logging.INFO)
logging.getLogger("urllib").setLevel(logging.INFO)

from verticox.vantage6 import (
    no_op,
    column_names,
    run_datanode,
    run_java_server,
    test_sum_local_features,
    fit, cross_validate
)
