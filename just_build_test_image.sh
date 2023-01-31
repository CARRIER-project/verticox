#!/bin/bash

TAG="harbor.carrier-mu.src.surf-hosted.nl/carrier/verticox:DEV"

echo "Building with tag" $TAG

docker build -t $TAG .
