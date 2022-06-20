#!/usr/bin/env sh

# Directory structure of proto files needs to be the same as package structure in pythons

python -m grpc_tools.protoc -I ./proto/ --python_out=./python/ \
--grpc_python_out=./python/ ./proto/verticox/grpc/*.proto