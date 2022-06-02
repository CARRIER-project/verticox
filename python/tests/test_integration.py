import logging

logging.basicConfig(level=logging.DEBUG)
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

import grpc
from sksurv.datasets import load_whas500

from verticox.aggregator import Aggregator
from verticox.datanode import DataNode
from verticox.grpc.datanode_pb2_grpc import add_DataNodeServicer_to_server, DataNodeStub

_logger = logging.getLogger(__name__)

MAX_WORKERS = 1
PORT = 7777

DATA_LIMIT = 5


def get_test_dataset(limit=None):
    features, events = load_whas500()

    if limit:
        features = features.head(limit)
        events = events[:DATA_LIMIT]

    features = features.values.astype(float)

    return features, events


def run_datanode_grpc_server(features, event_times, right_censored, port=PORT):
    server = grpc.server(ThreadPoolExecutor(max_workers=MAX_WORKERS))
    add_DataNodeServicer_to_server(DataNode(features=features, event_times=event_times,
                                            right_censored=right_censored), server)
    server.add_insecure_port(f'[::]:{port}')
    _logger.info(f'Starting datanode on port {port}')
    server.start()
    return server, port


def split_events(events):
    df = pd.DataFrame(events)
    times = df.lenfol.values
    right_censored = df.fstat.values

    return times, right_censored


# TODO: use multiple institutions
def test_integration(caplog):
    caplog.set_level(logging.DEBUG)

    features, events = get_test_dataset(limit=DATA_LIMIT)
    event_times, right_censored = split_events(events)
    server, port = run_datanode_grpc_server(features, event_times, right_censored)

    logging.info(f'Connecting to datanode at port {port}')
    channel = grpc.insecure_channel(f'localhost:{port}')
    stub = DataNodeStub(channel)
    institutions = [stub]

    logging.info(f'Initializing aggregator connected to {len(institutions)} institutions')
    aggregator = Aggregator([stub], event_times, right_censored)

    aggregator.fit()

    logging.info(f'Betas: {aggregator.get_betas()}')
