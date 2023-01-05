#!/usr/bin/env python3


import clize
import grpc

from verticox.grpc.datanode_pb2 import Empty
from verticox.grpc.datanode_pb2_grpc import DataNodeStub


def main(datanode_address, cert_chain):
    with open(cert_chain, 'rb') as f:
        creds = grpc.ssl_channel_credentials(f.read())
    channel = grpc.secure_channel(datanode_address, creds)
    stub = DataNodeStub(channel)

    feature_names = stub.getFeatureNames(Empty())

    print(feature_names)


if __name__ == "__main__":
    clize.run(main)
