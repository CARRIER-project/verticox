## Building
The makefile contains the necessary commands to build the verticox+ package.

If you wish to build the vantage6 docker image from scratch, use:
```
make docker
```

## Testing
There are several types of tests in this repository:
- Unit tests
- Integration test
- vantage6 tests

### Unit tests
The unit tests for the python components can be found in `python/tests` and can be run with pytest.
The unit tests for the java components are found in `java/src/test` and can be run with maven.

### Integration test
The files for the integration test can be found in `integration/`. It contains configuration files
for the java components, a `data` directory with a small dataset, a docker-compose file to run all
components, and a .env-example file that can be used to create your specific .env file. The .env
file is required to indicate the absolute path to the directory that contains the jar file for the
java components after you built them with maven.

The integration test can be run with `docker compose up`.

### Vantage6 tests
The main vantage6 test is the script `python/tests/test_verticox_v6.py`. This can be run as a
command-line tool and has parameters to configure the test to your needs:

```bash
./test_verticox_v6.py --help
Usage: test_verticox_v6.py [OPTIONS] host port user password

Arguments:
  host
  port
  user
  password

Options:
  --image=STR          (default: harbor2.vantage6.ai/algorithms/verticox:latest)
  --method=STR         (default: fit)
  --private-key=STR

Other actions:
  -h, --help          Show the help
```

You could run the test on a [local vantage6 network](https://docs.vantage6.ai/en/main/server/use
.html#testing). You have to make sure your nodes have access to data that is suitable for survival
analysis.

```bash
./test_verticox_v6.py http://localhost 7601 dev_admin password
```

## Further development
