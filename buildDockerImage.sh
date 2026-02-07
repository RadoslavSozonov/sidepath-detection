#!/bin/bash
set -e

REGISTRY=registry.digitalocean.com/scenwise-registry
IMAGE_NAME=sidewalk-api

docker build --no-cache -t ${IMAGE_NAME}:latest .
docker tag ${IMAGE_NAME}:latest ${REGISTRY}/${IMAGE_NAME}:main-latest
