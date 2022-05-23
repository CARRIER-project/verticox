#!/usr/bin/env sh

python -m grpc_tools.protoc -I ./proto/ --python_out=./python/verticox/grpc \
--grpc_python_out=./python/verticox/grpc ./proto/datanode.proto