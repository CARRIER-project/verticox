[![Tests](https://github.com/CARRIER-project/verticox/actions/workflows/tests.yml/badge.svg)](https://github.com/CARRIER-project/verticox/actions/workflows/tests.yml)
# Python Verticox implementation

This code implements the Verticox algorithm from [Dai et al., 2022](https://ieeexplore.ieee.org/document/9076318)
for use in Vantage6.

## Running tests locally

### Install `poetry`
First, ensure that you have `poetry` installed. See the [official instructions](https://python-poetry.org/docs/#installation).

### Install the package using poetry

```python
poetry install
```

### Run the integration tests

```python
poetry run python tests/test_integration.py
```
