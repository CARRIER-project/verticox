# Suppress numba debug logging
import logging
import sys

logging.basicConfig(stream=sys.stdout)
logging.getLogger('numba').setLevel(logging.INFO)
logging.getLogger('urllib').setLevel(logging.INFO)

from verticox.vantage6 import RPC_no_op, RPC_column_names, RPC_run_datanode, \
    RPC_run_java_server, RPC_test_sum_local_features, verticox
