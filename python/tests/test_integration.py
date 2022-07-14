import logging
import sys
logging.basicConfig(level=logging.DEBUG, handlers=[logging.FileHandler('log.txt', mode='w'),
                                                   logging.StreamHandler(sys.stdout)])
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


# def test_integration_one_institution():
#     _logger.addHandler(RotatingFileHandler('log.txt'))
#
#     features, events = get_test_dataset(limit=DATA_LIMIT, censored=False)
#
#     features = features.astype(np.float128)
#
#     target_result = get_target_result(features, events)
#
#     _logger.info(f'Target result: {target_result}')
#
#     num_features = features.shape[1]
#
#     event_times, right_censored = split_events(events)
#
#     p1 = Process(target=run_datanode,
#                  args=(event_times, features, right_censored, PORT1, 'first'))
#     try:
#         p1.start()
#
#         aggregator_process = Process(target=run_aggregator,
#                                      args=([PORT1], event_times, right_censored))
#
#         _logger.info('Starting aggregator')
#         aggregator_process.start()
#         aggregator_process.join()
#
#     except Exception as e:
#         traceback.print_exc()
#     finally:
#         # Make sure all processes are always killed
#         p1.kill()
#         p1.join()


def test_integration():
    features, events = get_test_dataset(limit=DATA_LIMIT, censored=False)

    features = features.astype(np.float128)

    target_result = get_target_result(features, events)

    _logger.info(f'Target result: {target_result}')

    num_features = features.shape[1]
    feature_split = num_features // 2
    features1 = features[:, :feature_split]
    features2 = features[:, feature_split:]
    event_times, right_censored = split_events(events)

    p1 = Process(target=run_datanode,
                 args=(event_times, features1, right_censored, PORT1, 'first'))
    p2 = Process(target=run_datanode, args=(event_times, features2, right_censored, PORT2,
                                            'second'))
    try:
        p1.start()
        p2.start()

        aggregator_process = Process(target=run_aggregator,
                                     args=([PORT1, PORT2], event_times, right_censored))

        _logger.info('Starting aggregator')
        aggregator_process.start()
        aggregator_process.join()

    except Exception as e:
        traceback.print_exc()
    finally:
        # Make sure all processes are always killed
        p1.kill()
        p2.kill()
        p1.join()
        p2.join()


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
