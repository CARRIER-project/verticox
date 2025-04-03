    FROM eclipse-temurin:17.0.14_7-jdk as runner

ARG PKG_NAME="verticox"
ENV JAR_PATH="/app/verticox.jar"
ENV PATH="/root/miniconda3/bin:${PATH}"

EXPOSE 8888
LABEL p8888=python
EXPOSE 9999
LABEL p9999=java

RUN apt update && apt install -y g++ musl-dev libffi-dev libssl-dev git

# Install Miniconda on x86 or ARM platforms
RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$arch" = "aarch64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
    echo "Unsupported architecture: $arch"; \
    exit 1; \
    fi && \
    wget $MINICONDA_URL -O miniconda.sh && \
    mkdir -p /root/.conda && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm -f miniconda.sh

RUN conda create -n verticox python=3.10
ENV PATH="/root/miniconda3/envs/verticox/bin:${PATH}"

COPY java/verticox/target/verticox*.jar $JAR_PATH

# install federated algorithm
COPY python/ /app/

WORKDIR /app
RUN pip install setuptools
RUN pip install -e .


ENV PKG_NAME=${PKG_NAME}

CMD python -c "from vantage6.algorithm.tools.wrap import wrap_algorithm; wrap_algorithm()"
