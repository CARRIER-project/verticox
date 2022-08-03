# basic python3 image as base
FROM continuumio/miniconda3:4.12.0

# This is a placeholder that should be overloaded by invoking
# docker build with '--build-arg PKG_NAME=...'
ARG PKG_NAME="verticox"

# install federated algorithm
COPY . /app

WORKDIR /app

RUN pip install poetry

RUN poetry install --no-root --no-dev

# In Vantage6 versions 3.1+, you can use VPN communication between algorithms
# over multiple ports. You can specify the ports that are allowed for
# communication here, along with a label that helps you identify them.
# If no ports are specified (and VPN is available), only port 8888 is available
# by default
EXPOSE 8888

ENV PKG_NAME=${PKG_NAME}

# Tell docker to execute `docker_wrapper()` when the image is run.
CMD poetry run python -c "from vantage6.tools.docker_wrapper import docker_wrapper; docker_wrapper('${PKG_NAME}')"