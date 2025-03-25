
IMAGE_NAME ?= harbor.carrier-mu.src.surf-hosted.nl/carrier/verticox
IMAGE_TAG ?=
MVN_ARGS =
MVN_SETTINGS ?=

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

python:
	pip install -e python/

python-docs-deps:
	pip install -e "python[docs]"

docs: python-docs-deps
	mkdocs build

deploy-docs: python-docs-deps
	mkdocs gh-deploy --force


docker: java
	docker build -t $(image) .

clean:
	cd java/verticox && mvn clean

serve-docs:
	mkdocs serve