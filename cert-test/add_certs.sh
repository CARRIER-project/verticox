#!/usr/bin/env bash

# Add root certificate to certificate store
cp myCA.pem /usr/local/share/ca-certificates/myCA.cert
update-ca-certificates
