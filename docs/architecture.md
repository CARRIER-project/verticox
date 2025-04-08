# Architecture

The most convenient way to run verticox is using the vantage6 framework, although it is possible to
run it without it.

The architecture of verticox+ includes both components written in Java and Python. The Java
components
encompass everything related to the n-party scalar product protocol. In contrast, the Python
components
are responsible for coordination and the rest of the verticox+ algorithm.

The architecture below shows how these components are divided. All nodes are intended to be run on
different machines.

![architecture](verticox_architecture.svg)
/// caption
The verticox+ high-level architecture
///

## Main python components

The python components in the architecture correspond to different modules in the codebase.

### `aggregator.py`

The main coordinating component. The aggregator kicks off the Cox Proportional hazards analysis
and provides the other components with the right starting parameters. The aggregator has access
to the outcome dataset.

### `datanode.py`

The datanodes hold the feature data. They are idle until the aggregator sends them a request over
[gRPC](https://grpc.io/).

### `nodemanager.py`
Nodemanager was created to provide a unified interface for running the algorithm with or without vantage6.
It is responsible for starting the aggregator and datanodes.

### Running Verticox+ without vantage6
It is possible to run verticox+ without vantage6. For example, the integration test that can be found
in the folder `integration/` runs the components inside a docker-compose environment without vantage6.

## Java components

The java components are responsible for the n-party scalar product protocol. This protocol is used
at the start of the Cox proportional hazards analysis to compute \( \beta_{k}^{p} \) in the Verticox
algorithm. More info on this is available at the [index page](index.md).

The java components can be found in the `java` folder. 

### Verticox N-party scalar product protocol wrapper

This is a wrapper that can be used to incorperate the N-party scalar product protocol within verticox by Dai et al. The
advantage of using this wrapper is that you no longer need to share the outcome among all parties. Instead this wrapper
will calculate the value in a privacy preserving manner. See associated powerpoint in this repositories for a more
detailed explanation.

It is possible for party A to contain values for individual 1, party B for individual 2 and party C knows the selection
criteria. It is also acceptable if A knows both the relevant values & selection criteria. It can als be the case that
both A & B know relevant values due to a hybrid split. In this case missing records need to have a value of ?. E.g. A
has record 1,2,3 B has record 4,5,6 then the data needs to look as follows: A = [A1,A2,A3,?,?,?] B= [?,?,?,B1,B1,B1]
Similarly the attribute used for the selection criteria may be split up as well.

To use this method the following needs to be done:

1) setPrecision (default 5)
2) sumRelevantValues
3) postZ
4) sumZvalues

### Implemented methods:

#### setPrecision:

Set the precision to be used for double values for the product protocol. Expected input:

- An int indicating the precision

Default precision used is 5. Always make sure to keep the precision the same across the various parties involved. To
keep everything aligned use setPrecisionCentral

#### sumRelevantValues:

Sums the values based on the relevant individuals. Expected input:

- Predictor: the predicator to be summed
- Requirements: list of attributeRequirement indicating the relevant individuals

Example input:

```json
{
  "requirements": [
    {
      "value": {
        "type": "numeric",
        "value": "1",
        "attributeName": "x6",
        "id": null,
        "uknown": false
      },
      "range": false,
      "upperLimit": null,
      "lowerLimit": null,
      "name": "x6"
    }
  ],
  "endpoint": "2"
}
```

#### postZ:

Updates the local z-values. Expected input:

- z: array of doubles representing the z values
- endpoint: name of the relevant server
-

#### sumZValues:

Sums the Z values based on the relevant individuals. Expected input:

- endpoint: the endpoint containing the relevant z values
- Requirements: list of attributeRequirement indicating the relevant individuals

#### cross-fold validation

It is possible to include k-fold crossvalidation. This can be done by setting the active records (i.e. the records
present in the current fold) via the "activateFold" endpoint. The request looks as follows:

```
{
  "activeRecords" : [ true, true, true, true, true, true, true, false, false, false ]
}
```

It is important to note that user should never have the ability to manually generate folds. If folds can be manually
generated it becomes possible to deduce the true data by repeatedly quering (e.g. set only 1 record to active and you
get that specific value). As such the libraries should be centrally generated, outside of the control of any potential
users. However, using this function properly is something that needs to covered by projects using this library.

#### SOAPUI example project

A SOAPUI example project is also present in this repository.

#### Download jar from github:

A maven settings.xml is included in this project that contains a bot for downloading jar files from github packages.
Unfortunatly it is not possible to download packages without a valid token, even when the project is public