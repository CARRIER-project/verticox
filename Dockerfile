FROM openjdk:17-slim as runner

# This is a placeholder that should be overloaded by invoking
# docker build with '--build-arg PKG_NAME=...'
ARG PKG_NAME="verticox"
ENV SERVER_PORT=8888

RUN apt update && apt install -y python3 python3-pip python3-dev g++ musl-dev libffi-dev \
    libssl-dev git
RUN ln -sf python3 /usr/bin/python
RUN pip3 install --no-cache setuptools wheel poetry

COPY java/verticox/target/verticox*.jar $JAR_PATH

# install federated algorithm
COPY python/ /app/

WORKDIR /app
RUN poetry install
RUN poetry run pip install pip install git+https://github.com/CARRIER-project/vantage6.git@337-feature-request-docker-wrapper-for-parquet-files#subdirectory=vantage6-common
RUN poetry run pip install pip install git+https://github.com/CARRIER-project/vantage6.git@337-feature-request-docker-wrapper-for-parquet-files#subdirectory=vantage6-client

ENV PKG_NAME=${PKG_NAME}

# Tell docker to execute `docker_wrapper()` when the image is run.
CMD poetry run python -c "from vantage6.tools.docker_wrapper import parquet_wrapper; parquet_wrapper('${PKG_NAME}')"
