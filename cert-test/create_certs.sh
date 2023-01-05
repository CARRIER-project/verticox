#!/usr/bin/env bash

CA_NAME="myCA"
SERVER_NAME="server"

echo "Creating root certificate key"
# CA root certificate key
openssl genrsa -out $SERVER_NAME.key 2048

echo "Creating root certificate"
# CA root certificate
openssl req -x509 -new -nodes -key $CA_NAME.key -sha256 -days 1825 -out $CA_NAME.pem

echo "Creating server key"

# Server certificate
openssl genrsa -out $SERVER_NAME.key 2048

echo "Creating server certificate request"

# Signing request
openssl req -new -key $SERVER_NAME.key -out $SERVER_NAME.csr

echo "Create signed certificate from request"

openssl x509 -req -in $SERVER_NAME.csr -CA $CA_NAME.pem -CAkey $CA_NAME.key \
-CAcreateserial -out $SERVER_NAME.crt -days 825 -sha256 -extfile $SERVER_NAME.ext


echo "Done"