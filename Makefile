
IMAGE_NAME ?= harbor2.vantage6.ai/carrier/verticox
IMAGE_TAG ?=
MVN_ARGS =
MVN_SETTINGS ?=
DOCKER_ARGS ?=

JARFILE = java/verticox/target/verticox-1.0-SNAPSHOT.jar



ifeq ($(IMAGE_TAG), )
	image := $(IMAGE_NAME)
else
	image := $(IMAGE_NAME):$(IMAGE_TAG)
endif

ifneq ($(MVN_SETTINGS), )
	MVN_ARGS := --settings $(MVN_SETTINGS)
endif

$(JARFILE):
	cd java/verticox && mvn  $(MVN_ARGS) package

java: $(JARFILE)

.PHONY: python

python: python/pyproject.toml
	pip install -e python/

python-docs-deps:
	pip install -e "python[docs]"

docs: python-docs-deps
	mkdocs build

deploy-docs: python-docs-deps
	mkdocs gh-deploy --force

.PHONY: docker

docker: java
	docker build $(DOCKER_ARGS) -t $(image) .

clean:
	cd java/verticox && mvn clean
	rm -rf python/__pycache__

serve-docs:
	mkdocs serve