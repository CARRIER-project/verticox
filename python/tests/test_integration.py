import logging
from concurrent.futures import ThreadPoolExecutor

import grpc
from sksurv.datasets import load_whas500

from verticox.aggregator import Aggregator
from verticox.datanode import DataNode
from verticox.grpc.datanode_pb2_grpc import add_DataNodeServicer_to_server, DataNodeStub

_logger = logging.getLogger(__name__)

MAX_WORKERS = 1
PORT = 8888


def get_test_dataset():
    features, events = load_whas500()
    features = features.values.astype(float)

    return features, events


def run_datanode_grpc_server(port=PORT):
    features, events = get_test_dataset()
    server = grpc.server(ThreadPoolExecutor(max_workers=MAX_WORKERS))
    add_DataNodeServicer_to_server(DataNode(features=features, events=events), server)
    server.add_insecure_port(f'[::]:{port}')
    _logger.info(f'Starting datanode on port {port}')
    server.start()
    return server, port


# TODO: use multiple institutions
def test_integration(caplog):
    caplog.set_level(logging.DEBUG)
    server, port = run_datanode_grpc_server()

    logging.info(f'Connecting to datanode at port {port}')
    channel = grpc.insecure_channel(f'localhost:{port}')
    stub = DataNodeStub(channel)
    institutions = [stub]

    logging.info(f'Initializing aggregator connected to {len(institutions)} institutions')
    aggregator = Aggregator([stub])

    aggregator.fit()
