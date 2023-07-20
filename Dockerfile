FROM eclipse-temurin:17.0.7_7-jdk as runner

ARG PKG_NAME="verticox"
ENV JAR_PATH="/app/verticox.jar"

EXPOSE 8888
LABEL p8888=python
EXPOSE 9999
LABEL p9999=java

RUN apt update && apt install -y python3 python3-pip python3-dev g++ musl-dev libffi-dev \
    libssl-dev git
RUN ln -sf python3 /usr/bin/python
RUN pip3 install --no-cache setuptools wheel poetry

COPY java/verticox/target/verticox*.jar $JAR_PATH

# install federated algorithm
COPY python/ /app/

WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/pypoetry/cache \
    --mount=type=cache,target=/root/.cache/pypoetry/artifacts \
    poetry install


ENV PKG_NAME=${PKG_NAME}

# Tell docker to execute `docker_wrapper()` when the image is run.
CMD poetry run python -c "from vantage6.tools.docker_wrapper import auto_wrapper; auto_wrapper('${PKG_NAME}')"
