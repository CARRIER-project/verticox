# Testing
There are several types of tests in this repository:
- Unit tests
- Integration test
- vantage6 tests

## Unit tests
The unit tests for the python components can be found in `python/tests` and can be run with pytest.
The unit tests for the java components are found in `java/src/test` and can be run with maven.

## Integration test
The files for the integration test can be found in `integration/`. It contains configuration files
for the java components, a `data` directory with a small dataset, a docker-compose file to run all
components, and a .env-example file that can be used to create your specific .env file. The .env
file is required to indicate the absolute path to the directory that contains the jar file for the
java components after you built them with maven.

The integration test can be run with `docker compose up`.