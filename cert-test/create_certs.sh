#!/usr/bin/env bash

# CA root certificate key
openssl genrsa -out myCA.key 2048

echo "Created root certificate key"

# CA root certificate
openssl req -x509 -new -nodes -key myCA.key -sha256 -days 1825 -out myCA.pem

echo "Created root certificate"

# Server certificate
openssl genrsa -out server.key 2048

echo "Created server certificate"

# Signing request
openssl req -new -key server.key -out server.csr

echo "Created CSR"