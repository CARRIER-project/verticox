import json
import logging
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Array
from typing import List

import grpc
import numpy as np
import pandas as pd
from numba import types
from numpy.typing import ArrayLike
from sksurv.datasets import load_whas500
from sksurv.linear_model import CoxPHSurvivalAnalysis

from verticox.aggregator import Aggregator
from verticox.datanode import DataNode
from verticox.grpc.datanode_pb2 import Empty
from verticox.grpc.datanode_pb2_grpc import add_DataNodeServicer_to_server, DataNodeStub

logging.basicConfig(level=logging.DEBUG, handlers=[logging.FileHandler('log.txt', mode='w'),
                                                   logging.StreamHandler(sys.stdout)])
_logger = logging.getLogger(__name__)

MAX_WORKERS = 5
ROW_LIMIT = 5  # Number of samples to use
FEATURE_LIMIT = 2  # Number of features to use
RIGHT_CENSORED = True  # Whether to include right censored data
NUM_INSTITUTIONS = 2  # Number of institutions to split the data over
FIRST_PORT = 7777
PORTS = tuple(range(FIRST_PORT, FIRST_PORT + NUM_INSTITUTIONS))
DECIMAL_PRECISION = 3  # The precision to use when comparing results to target results
CONVERGENCE_PRECISION = 1e-4  # When difference between outer iterations falls below this, stop
NEWTON_RAPHSON_PRECISION = 1e-4  # Stopping condition for Newton-Raphson (epsilon)


def get_test_dataset(limit=None, feature_limit=None, include_right_censored=True):
    features, events = load_whas500()

    if not include_right_censored:
        features = features[uncensored(events)]
        events = events[uncensored(events)]
    if include_right_censored and limit:
        # Make sure there's both right censored and non-right censored data
        # Since the behavior should be deterministic we will still just take the first samples we
        # that meets the requirements.
        non_censored = uncensored(events)
        non_censored_idx = np.argwhere(non_censored).flatten()
        right_censored_idx = np.argwhere(~non_censored).flatten()

        limit_per_type = limit // 2

        non_censored_idx = non_censored_idx[:limit_per_type]
        right_censored_idx = right_censored_idx[:(limit - limit_per_type)]

        all_idx = np.concatenate([non_censored_idx, right_censored_idx])

        events = events[all_idx]
        features = features.iloc[all_idx]

    numerical_columns = features.columns[features.dtypes == float]

    features = features[numerical_columns]

    if limit:
        features = features.head(limit)
        events = events[:limit]

    features = features.values.astype(float)
    if feature_limit:
        features = features[:, :feature_limit]
    return features, events


def run_datanode_grpc_server(features, event_times, right_censored, port, name):
    server = grpc.server(ThreadPoolExecutor(),
                         )
    add_DataNodeServicer_to_server(DataNode(features=features, event_times=event_times,
                                            event_happened=right_censored, name=name,
                                            server=server), server)
    server.add_insecure_port(f'[::]:{port}')
    _logger.info(f'Starting datanode on port {port}')
    server.start()
    server.wait_for_termination()


def split_events(events):
    df = pd.DataFrame(events)
    times = df.lenfol.values
    right_censored = df.fstat.values

    return times, right_censored


def get_target_result(features, events):
    model = CoxPHSurvivalAnalysis()

    model.fit(features, events)

    return model.coef_


@np.vectorize
def uncensored(event):
    return event[0]


def integration_test(ports=PORTS, row_limit=ROW_LIMIT, feature_limit=FEATURE_LIMIT,
                     right_censored=True, convergence_precision=CONVERGENCE_PRECISION,
                     newton_raphson_precision=NEWTON_RAPHSON_PRECISION):
    num_institutions = len(ports)
    features, events = get_test_dataset(limit=row_limit,
                                        feature_limit=feature_limit,
                                        include_right_censored=right_censored)

    target_result = get_target_result(features, events)

    num_features = features.shape[1]
    feature_split = num_features // num_institutions

    _logger.info(f'Target result: {json.dumps(target_result.tolist())}')

    features_per_institution = list(chunk_features(feature_split, features))

    event_times, right_censored = split_events(events)

    processes = create_processes(event_times, features_per_institution, right_censored, ports)
    processes = list(processes)

    try:
        for p in processes:
            p.start()

        result = Array('d', features.shape[1])
        aggregator_process = Process(target=run_aggregator,
                                     args=(ports, event_times, right_censored,
                                           convergence_precision, newton_raphson_precision,
                                           result))

        _logger.info('Starting aggregator')
        aggregator_process.start()
        aggregator_process.join()

        np.testing.assert_array_almost_equal(np.array(result), target_result,
                                             decimal=DECIMAL_PRECISION)

    except Exception as e:
        traceback.print_exc()
    finally:
        # Make sure all processes are always killed
        for p in processes:
            p.join()
            p.kill()


def chunk_list(feature_split, target_result):
    for i in range(0, len(target_result), feature_split):
        yield target_result[i:i + feature_split].tolist()


def chunk_features(feature_split, features):
    for i in range(0, features.shape[1], feature_split):
        yield features[:, i:i + feature_split]


def create_processes(event_times, features_per_institution, right_censored, ports):
    for idx, f in enumerate(features_per_institution):
        p = Process(target=run_datanode,
                    args=(event_times, f, right_censored, ports[idx], f'institution no. {idx}'))

        yield p


def run_aggregator(ports: List[int], event_times: types.float64, right_censored: types.boolean[:],
                   convergence_precision: float, newton_raphson_precision: float,
                   result: Array):
    stubs = [get_datanode_client(port) for port in ports]

    aggregator = Aggregator(stubs, event_times, right_censored,
                            convergence_precision=convergence_precision,
                            newton_raphson_precision=newton_raphson_precision)

    start_time = time.time()
    aggregator.fit()

    _logger.info(f'Converged after {time.time() - start_time} seconds')

    betas = aggregator.get_betas().tolist()

    # I need to flatten the array
    betas = [el for sublist in betas for el in sublist]

    for i in range(len(betas)):
        result[i] = betas[i]

    _logger.info(f'Resulting betas: {json.dumps(aggregator.get_betas().tolist())}')

    for stub in stubs:
        stub.kill(Empty())


def run_datanode(event_times, features, right_censored, port, name):
    run_datanode_grpc_server(features, event_times, right_censored, port, name)


def get_datanode_client(port):
    logging.info(f'Connecting to datanode at port {port}')
    channel = grpc.insecure_channel(f'localhost:{port}')
    # ready = grpc.channel_ready_future(channel)
    # ready.result(timeout=300)
    stub = DataNodeStub(channel)
    return stub


class LzFilter(logging.Filter):

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if message.startswith('Lz: '):
            value = message[4:]
            print('bla')
            record.lz_value = value
            return True
        return False


if __name__ == '__main__':
    row_limit = ROW_LIMIT
    feature_limit = FEATURE_LIMIT
    right_censored = RIGHT_CENSORED
    integration_test(PORTS, ROW_LIMIT, FEATURE_LIMIT, RIGHT_CENSORED)
