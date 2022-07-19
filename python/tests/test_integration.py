import json
import logging
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process

import grpc
import numpy as np
import pandas as pd
from sksurv.datasets import load_whas500
from sksurv.linear_model import CoxPHSurvivalAnalysis

from verticox.aggregator import Aggregator
from verticox.datanode import DataNode
from verticox.grpc.datanode_pb2_grpc import add_DataNodeServicer_to_server, DataNodeStub

logging.basicConfig(level=logging.DEBUG, handlers=[logging.FileHandler('log.txt', mode='w'),
                                                   logging.StreamHandler(sys.stdout)])
_logger = logging.getLogger(__name__)

MAX_WORKERS = 5
PORT1 = 7777
PORT2 = 7779
GRPC_OPTIONS = [('wait_for_ready', True)]
DATA_LIMIT = 100


def get_test_dataset(limit=None, censored=True):
    features, events = load_whas500()

    if not censored:
        features = features[include(events)]
        events = events[include(events)]

    numerical_columns = features.columns[features.dtypes == float]

    features = features[numerical_columns]

    if limit:
        features = features.head(limit)
        events = events[:limit]

    features = features.values.astype(float)

    return features, events


def run_datanode_grpc_server(features, event_times, right_censored, port, name):
    server = grpc.server(ThreadPoolExecutor(),
                         options=GRPC_OPTIONS)
    add_DataNodeServicer_to_server(DataNode(features=features, event_times=event_times,
                                            right_censored=right_censored, name=name), server)
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
def include(event):
    return event[0]


def test_integration(ports=(PORT1, PORT2)):
    num_institutions = len(ports)
    features, events = get_test_dataset(limit=DATA_LIMIT, censored=False)

    # scaler = StandardScaler()
    # features = scaler.fit_transform(X=features)

    target_result = get_target_result(features, events)

    num_features = features.shape[1]
    feature_split = num_features // num_institutions

    splitted_target = list(chunk_list(feature_split, target_result))

    _logger.info(f'Target result: {json.dumps(splitted_target)}')

    features_per_institution = list(chunk_features(feature_split, features))

    event_times, right_censored = split_events(events)

    processes = create_processes(event_times, features_per_institution, right_censored, ports)
    processes = list(processes)

    try:
        for p in processes:
            p.start()

        aggregator_process = Process(target=run_aggregator,
                                     args=(ports, event_times, right_censored))

        _logger.info('Starting aggregator')
        aggregator_process.start()
        aggregator_process.join()

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


def run_aggregator(ports, event_times, right_censored):
    stubs = [get_datanode_client(port) for port in ports]

    aggregator = Aggregator(stubs, event_times, right_censored)

    aggregator.fit()

    _logger.info(f'Resulting betas: {aggregator.get_betas()}')


def run_datanode(event_times, features, right_censored, port, name):
    run_datanode_grpc_server(features, event_times, right_censored, port, name)


def get_datanode_client(port):
    logging.info(f'Connecting to datanode at port {port}')
    channel = grpc.insecure_channel(f'localhost:{port}', options=GRPC_OPTIONS)
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
    test_integration()
